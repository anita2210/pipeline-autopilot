"""
rag_chatbot.py
--------------
RAG Chatbot using tiered knowledge base + Gemini for failure diagnosis.

Functions:
- get_day_stats(day)
- get_hour_stats(hour)
- get_similar_runs(features)
- get_top_failure_type(similar_runs)
- get_diagnosis(pipeline_features, failure_prob, user_message, chat_history)

Author  : Member 3 (AI Engineer)
Project : Pipeline Autopilot — CI/CD Failure Prediction System
Date    : April 2026
"""

import json
import pickle
import logging
import os
from pathlib import Path

import numpy as np
import faiss
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag_chatbot")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
KB_DIR = Path(__file__).resolve().parents[1] / "knowledge_base"

# ---------------------------------------------------------------------------
# Load knowledge base at startup
# ---------------------------------------------------------------------------
def _load_json(filename):
    path = KB_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {path}")
    with open(path) as f:
        return json.load(f)

def _load_faiss():
    path = KB_DIR / "similar_runs_index.pkl"
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

logger.info("Loading knowledge base...")
GLOBAL_STATS  = _load_json("global_stats.json")
DAILY_STATS   = _load_json("daily_stats.json")
REPO_STATS    = _load_json("repo_stats.json")
ERROR_STATS   = _load_json("error_stats.json")
FAISS_DATA    = _load_faiss()
logger.info("Knowledge base loaded successfully.")

# ---------------------------------------------------------------------------
# Gemini Setup
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("Gemini configured successfully.")
else:
    client = None
    logger.warning("GEMINI_API_KEY not set — chatbot will use fallback responses.")

# ---------------------------------------------------------------------------
# Stats functions
# ---------------------------------------------------------------------------

def get_day_stats(day: int) -> dict:
    """Get failure stats for a given day of week (0=Mon, 6=Sun)."""
    return DAILY_STATS["by_day_of_week"].get(str(day), {})


def get_hour_stats(hour: int) -> dict:
    """Get failure stats for a given hour (0-23)."""
    return DAILY_STATS["by_hour"].get(str(hour), {})


def get_similar_runs(features: dict, top_k: int = 5) -> list:
    """
    Find top_k similar runs using FAISS index built from 149K training rows.
    Returns list of dicts with similarity info.
    """
    feature_cols = FAISS_DATA["feature_cols"]
    vector = np.array(
        [float(features.get(col, 0)) for col in feature_cols],
        dtype=np.float32
    ).reshape(1, -1)

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    # Handle different FAISS versions
    try:
        # New FAISS API
        distances, indices = FAISS_DATA["index"].search(vector, top_k)
    except TypeError:
        try:
            # Old FAISS API with output arrays
            distances = np.zeros((1, top_k), dtype=np.float32)
            indices   = np.zeros((1, top_k), dtype=np.int64)
            FAISS_DATA["index"].search_and_reconstruct(vector, top_k, distances, indices)
        except Exception:
            # Fallback: return first top_k labels directly
            n = min(top_k, len(FAISS_DATA["labels"]))
            distances = np.zeros((1, n), dtype=np.float32)
            indices   = np.arange(n, dtype=np.int64).reshape(1, n)
    labels        = FAISS_DATA["labels"]
    failure_types = FAISS_DATA["failure_types"]

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(labels):
            results.append({
                "index"       : int(idx),
                "distance"    : round(float(dist), 4),
                "failed"      : int(labels[idx]),
                "failure_type": str(failure_types[idx]),
            })
    return results


def get_top_failure_type(similar_runs: list) -> str:
    """Get most common failure type from similar runs."""
    failed_runs = [r for r in similar_runs if r["failed"] == 1]
    if not failed_runs:
        return "unknown"
    counts = {}
    for r in failed_runs:
        ft = r["failure_type"]
        counts[ft] = counts.get(ft, 0) + 1
    return max(counts, key=counts.get)


def get_global_context() -> dict:
    """Return a compact version of global stats for Gemini prompt."""
    return {
        "failure_rate"             : GLOBAL_STATS["failure_rate"],
        "avg_retry_count"          : GLOBAL_STATS["avg_retry_count"],
        "avg_failures_last_7_runs" : GLOBAL_STATS["avg_failures_last_7_runs"],
        "top_failure_types"        : list(GLOBAL_STATS["top_failure_types"].keys())[:3],
    }


# ---------------------------------------------------------------------------
# Gemini call with multi-turn support
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str, chat_history: list = []) -> str:
    """Call Gemini with full conversation history for multi-turn support."""
    if client is None:
        return "Gemini API key not configured. Set GEMINI_API_KEY environment variable."
    try:
        # Build history for multi-turn conversation
        history = []
        for turn in chat_history:
            role    = turn.get("role", "")
            content = turn.get("content", "")
            if role == "user":
                history.append(types.Content(role="user",  parts=[types.Part(text=content)]))
            elif role == "assistant":
                history.append(types.Content(role="model", parts=[types.Part(text=content)]))

        # Add current prompt
        history.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

        # Retry up to 3 times on 503 overload
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=history,
                )
                return response.text.strip()
            except Exception as e:
                if "503" in str(e) and attempt < 2:
                    import time
                    logger.warning("Gemini 503 — retrying in 3s (attempt %d/3)", attempt+1)
                    time.sleep(3)
                else:
                    raise e

    except Exception as e:
        logger.error("Gemini error: %s", e)
        return f"Gemini error: {str(e)}"


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    pipeline_features : dict,
    failure_prob      : float,
    day_stats         : dict,
    hour_stats        : dict,
    similar_runs      : list,
    top_failure_type  : str,
    global_ctx        : dict,
    user_message      : str,
) -> str:
    """Build a context-rich prompt grounded in 149K training data."""

    similar_fail_count = sum(1 for r in similar_runs if r["failed"] == 1)
    risk_level = "HIGH" if failure_prob > 0.75 else "MEDIUM" if failure_prob > 0.4 else "LOW"

    prompt = f"""You are Kairos Pulse, an expert CI/CD reliability engineer and MLOps assistant.
You have access to historical data from 149,000 real pipeline runs to ground your answers.

═══ LIVE PIPELINE CONTEXT ═══
- Pipeline        : {pipeline_features.get('pipeline_name', 'unknown')}
- Repo            : {pipeline_features.get('repo', 'unknown')}
- Branch          : {pipeline_features.get('head_branch', 'unknown')}
- Failure Prob    : {failure_prob:.1%}
- Risk Level      : {risk_level}
- Failure Rate    : {pipeline_features.get('workflow_failure_rate', 0):.1%}
- Failures last 7 : {pipeline_features.get('failures_last_7_runs', 0)}
- Prev run status : {str(pipeline_features.get('prev_run_status', 'unknown')).upper()}
- Retry count     : {pipeline_features.get('retry_count', 0)}
- Concurrent runs : {pipeline_features.get('concurrent_runs', 1)}

═══ HISTORICAL PATTERNS (from 149K training runs) ═══
- Global failure rate          : {global_ctx['failure_rate']}
- Avg retry count (global)     : {global_ctx['avg_retry_count']}
- Avg failures last 7 (global) : {global_ctx['avg_failures_last_7_runs']}
- Top failure types globally   : {', '.join(global_ctx['top_failure_types'])}
- Day-of-week failure rate     : {day_stats.get('failure_rate', 'N/A')}
- Hour-of-day failure rate     : {hour_stats.get('failure_rate', 'N/A')}
- Similar historical runs found: {len(similar_runs)} (of which {similar_fail_count} failed)
- Most common failure type     : {top_failure_type}

═══ USER QUESTION ═══
{user_message}

Answer conversationally like a senior engineer talking to a teammate.
Use markdown bold for key terms. Be specific and practical.
If risk is HIGH, always end with a clear one-line action recommendation.
Keep answer under 150 words unless a detailed breakdown is explicitly asked for."""

    return prompt


# ---------------------------------------------------------------------------
# Main RAG function
# ---------------------------------------------------------------------------

def get_diagnosis(
    pipeline_features : dict,
    failure_prob      : float,
    user_message      : str  = "Give me a full diagnosis of this pipeline run.",
    chat_history      : list = [],
) -> str:
    """
    Full RAG diagnosis for a pipeline run. Answers any user question
    grounded in 149K historical runs via FAISS similarity search.

    Parameters
    ----------
    pipeline_features : dict  — live run feature values
    failure_prob      : float — predicted failure probability
    user_message      : str   — what the user typed in the chatbot
    chat_history      : list  — previous turns [{"role":"user","content":"..."}]

    Returns
    -------
    str — Gemini's answer, ready to display in chat bubble
    """
    logger.info("Running RAG diagnosis (prob=%.4f, question='%s')...",
                failure_prob, user_message[:60])

    day  = int(pipeline_features.get("day_of_week", 0))
    hour = int(pipeline_features.get("hour", 12))

    day_stats    = get_day_stats(day)
    hour_stats   = get_hour_stats(hour)
    similar_runs = get_similar_runs(pipeline_features)
    top_ft       = get_top_failure_type(similar_runs)
    global_ctx   = get_global_context()

    prompt = _build_prompt(
        pipeline_features, failure_prob,
        day_stats, hour_stats,
        similar_runs, top_ft, global_ctx,
        user_message,
    )

    response = _call_gemini(prompt, chat_history=chat_history)
    logger.info("Diagnosis complete.")
    return response


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_features = {
        "pipeline_name"        : "deploy-staging",
        "repo"                 : "ClickHouse/ClickHouse",
        "head_branch"          : "backport/23.8",
        "day_of_week"          : 0,
        "hour"                 : 14,
        "retry_count"          : 3,
        "failures_last_7_runs" : 4,
        "duration_deviation"   : 2.5,
        "workflow_failure_rate": 0.6,
        "concurrent_runs"      : 8,
        "total_jobs"           : 12,
        "failed_jobs"          : 5,
        "duration_seconds"     : 450,
        "prev_run_status"      : "failure",
    }

    questions = [
        "Give me a full diagnosis.",
        "How do I fix it?",
        "Is it safe to run?",
        "What is the biggest red flag here?",
    ]

    history = []
    print("\n" + "="*60)
    print("RAG CHATBOT TEST — multi-turn")
    print("="*60)

    for q in questions:
        print(f"\nUser: {q}")
        answer = get_diagnosis(test_features, 0.85, user_message=q, chat_history=history)
        print(f"Bot : {answer}")
        history.append({"role": "user",      "content": q})
        history.append({"role": "assistant", "content": answer})

    print("="*60)