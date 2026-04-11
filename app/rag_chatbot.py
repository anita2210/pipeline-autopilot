"""
rag_chatbot.py
--------------
RAG Chatbot using tiered knowledge base + Gemini for failure diagnosis.

Functions:
- get_day_stats(day)
- get_hour_stats(hour)
- get_similar_runs(features)
- get_top_failure_type(similar_runs)
- get_diagnosis(pipeline_features, failure_prob)

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
from openai import OpenAI

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
# Load knowledge base at startup (no pandas at runtime)
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
# OpenAI Setup
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI configured successfully.")
else:
    openai_client = None
    logger.warning("OPENAI_API_KEY not set.")
# ---------------------------------------------------------------------------
# 5 Stats functions
# ---------------------------------------------------------------------------

def get_day_stats(day: int) -> dict:
    """Get failure stats for a given day of week (0=Mon, 6=Sun)."""
    return DAILY_STATS["by_day_of_week"].get(str(day), {})


def get_hour_stats(hour: int) -> dict:
    """Get failure stats for a given hour (0-23)."""
    return DAILY_STATS["by_hour"].get(str(hour), {})


def get_similar_runs(features: dict, top_k: int = 5) -> list:
    """
    Find top_k similar runs using FAISS index.
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

    distances, indices = FAISS_DATA["index"].search(vector, top_k)
    labels = FAISS_DATA["labels"]
    failure_types = FAISS_DATA["failure_types"]

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(labels):
            results.append({
                "index": int(idx),
                "distance": round(float(dist), 4),
                "failed": int(labels[idx]),
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
        "failure_rate": GLOBAL_STATS["failure_rate"],
        "avg_retry_count": GLOBAL_STATS["avg_retry_count"],
        "avg_failures_last_7_runs": GLOBAL_STATS["avg_failures_last_7_runs"],
        "top_failure_types": list(GLOBAL_STATS["top_failure_types"].keys())[:3],
    }


# ---------------------------------------------------------------------------
# OpenAI integration
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str) -> str:
    if openai_client is None:
        return "OpenAI not configured. Set OPENAI_API_KEY."
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("OpenAI error: %s", e)
        return f"OpenAI error: {str(e)}"

def _build_prompt(
    pipeline_features: dict,
    failure_prob: float,
    day_stats: dict,
    hour_stats: dict,
    similar_runs: list,
    top_failure_type: str,
    global_ctx: dict,
) -> str:
    """Build structured prompt for Gemini."""
    similar_fail_count = sum(1 for r in similar_runs if r["failed"] == 1)

    prompt = f"""You are an MLOps expert analyzing a CI/CD pipeline failure prediction.

PIPELINE CONTEXT:
- Failure probability: {failure_prob:.1%}
- Risk level: {"HIGH" if failure_prob > 0.75 else "MEDIUM" if failure_prob > 0.4 else "LOW"}
- Day of week failure rate: {day_stats.get('failure_rate', 'N/A')}
- Hour of day failure rate: {hour_stats.get('failure_rate', 'N/A')}
- Similar historical runs: {len(similar_runs)} found, {similar_fail_count} failed
- Most common failure type in similar runs: {top_failure_type}
- Global failure rate: {global_ctx['failure_rate']}
- Key features: retry_count={pipeline_features.get('retry_count', 0)}, failures_last_7_runs={pipeline_features.get('failures_last_7_runs', 0)}, duration_deviation={pipeline_features.get('duration_deviation', 0):.2f}

Based on this context, provide exactly 3 sections:

1. WHY: In 2-3 sentences, explain WHY this pipeline is likely to fail.
2. FIX: In 2-3 bullet points, provide specific actionable fixes.
3. CODE: Provide one short bash or yaml code snippet that would help address the issue.

Be specific and practical. Focus on the most likely failure type: {top_failure_type}."""

    return prompt


# ---------------------------------------------------------------------------
# Main RAG function
# ---------------------------------------------------------------------------

def get_diagnosis(pipeline_features: dict, failure_prob: float) -> dict:
    """
    Full RAG diagnosis for a pipeline run.

    Parameters
    ----------
    pipeline_features : dict of feature name -> value
    failure_prob      : float, predicted failure probability (0-1)

    Returns
    -------
    dict with:
        - risk_score
        - risk_level
        - day_pattern
        - hour_pattern
        - similar_runs_count
        - similar_runs_failed
        - top_failure_type
        - gemini_why
        - gemini_fix
        - gemini_code
        - full_gemini_response
    """
    logger.info("Running RAG diagnosis (prob=%.4f)...", failure_prob)

    day   = int(pipeline_features.get("day_of_week", 0))
    hour  = int(pipeline_features.get("hour", 12))

    day_stats    = get_day_stats(day)
    hour_stats   = get_hour_stats(hour)
    similar_runs = get_similar_runs(pipeline_features)
    top_ft       = get_top_failure_type(similar_runs)
    global_ctx   = get_global_context()

    prompt   = _build_prompt(
        pipeline_features, failure_prob,
        day_stats, hour_stats,
        similar_runs, top_ft, global_ctx
    )
    gemini_response = _call_gemini(prompt)

    # Parse Gemini response into sections
    why_text  = ""
    fix_text  = ""
    code_text = ""
    lines = gemini_response.split("\n")
    current = None
    for line in lines:
        if line.startswith("1. WHY") or line.startswith("WHY"):
            current = "why"
        elif line.startswith("2. FIX") or line.startswith("FIX"):
            current = "fix"
        elif line.startswith("3. CODE") or line.startswith("CODE"):
            current = "code"
        elif current == "why":
            why_text += line + " "
        elif current == "fix":
            fix_text += line + " "
        elif current == "code":
            code_text += line + "\n"

    result = {
        "risk_score"          : round(failure_prob, 4),
        "risk_level"          : "HIGH" if failure_prob > 0.75 else "MEDIUM" if failure_prob > 0.4 else "LOW",
        "day_pattern"         : day_stats,
        "hour_pattern"        : hour_stats,
        "similar_runs_count"  : len(similar_runs),
        "similar_runs_failed" : sum(1 for r in similar_runs if r["failed"] == 1),
        "top_failure_type"    : top_ft,
        "gemini_why"          : why_text.strip() or gemini_response,
        "gemini_fix"          : fix_text.strip(),
        "gemini_code"         : code_text.strip(),
        "full_gemini_response": gemini_response,
    }

    logger.info("Diagnosis complete. Risk: %s | Top failure: %s", result["risk_level"], top_ft)
    return result


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_features = {
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
    }

    result = get_diagnosis(test_features, failure_prob=0.85)

    print("\n" + "="*60)
    print("RAG DIAGNOSIS RESULT")
    print("="*60)
    print(f"Risk Score : {result['risk_score']}")
    print(f"Risk Level : {result['risk_level']}")
    print(f"Top Failure: {result['top_failure_type']}")
    print(f"Similar Runs: {result['similar_runs_count']} ({result['similar_runs_failed']} failed)")
    print(f"\nGemini Response:\n{result['full_gemini_response']}")
    print("="*60)
