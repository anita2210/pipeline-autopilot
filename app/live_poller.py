"""
live_poller.py
--------------
Polls the GitHub Actions API every 30 seconds for new workflow runs
across the monitored repos. Extracts features from each new run and
queues them for live prediction via the /stream SSE endpoint.

Author  : Member 4 (MLOps Monitor)
Project : Pipeline Autopilot — CI/CD Failure Prediction System
Date    : April 2026
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("live_poller")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_API_BASE = "https://api.github.com"
POLL_INTERVAL_SECONDS = 30
RISK_THRESHOLD = 0.75

# Repos to monitor — same ones used for training
MONITORED_REPOS = [
    "apache/airflow",
    "kubernetes/kubernetes",
    "tensorflow/tensorflow",
    "pytorch/pytorch",
    "apache/spark",
]

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(run: dict, repo: str) -> dict:
    """
    Extract prediction features from a GitHub Actions workflow run dict.

    Parameters
    ----------
    run  : Raw workflow run dict from GitHub API.
    repo : Repository full name (e.g. 'apache/airflow').

    Returns
    -------
    dict — Feature dict compatible with the trained model.
    """
    created_at = run.get("created_at", "")
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except Exception:
        dt = datetime.now(timezone.utc)

    # Duration in seconds if run is completed
    duration_seconds = 0
    if run.get("updated_at") and run.get("created_at"):
        try:
            updated = datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00"))
            created = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
            duration_seconds = max(0, (updated - created).total_seconds())
        except Exception:
            duration_seconds = 0

    trigger_type = run.get("event", "push")
    head_branch = run.get("head_branch", "main")
    run_attempt = run.get("run_attempt", 1)

    return {
        "run_id": str(run.get("id", "")),
        "pipeline_name": run.get("name", "unknown"),
        "repo": repo,
        "trigger_type": trigger_type,
        "head_branch": head_branch,
        "hour": dt.hour,
        "day_of_week": dt.weekday(),
        "is_weekend": int(dt.weekday() >= 5),
        "is_main_branch": int(head_branch in ("main", "master")),
        "is_bot_triggered": int("bot" in run.get("triggering_actor", {}).get("login", "").lower()
                                if run.get("triggering_actor") else 0),
        "retry_count": max(0, run_attempt - 1),
        "duration_seconds": duration_seconds,
        "total_jobs": run.get("jobs_count", 1),
        "failed_jobs": 0,           # filled after completion
        "concurrent_runs": 1,       # approximated
        # Historical features — use defaults (cold start)
        "prev_run_status": 0,
        "failures_last_7_runs": 0,
        "avg_duration_7_runs": duration_seconds,
        "duration_deviation": 0.0,
        "workflow_failure_rate": 0.0,
        "hours_since_last_run": 24,
        "is_first_run": 0,
    }


# ---------------------------------------------------------------------------
# GitHub API calls
# ---------------------------------------------------------------------------

async def fetch_recent_runs(
    client: httpx.AsyncClient,
    repo: str,
    since: Optional[str] = None,
    per_page: int = 5,
) -> list:
    """
    Fetch the most recent workflow runs for a repository.

    Parameters
    ----------
    client   : Async HTTP client.
    repo     : Repository full name.
    since    : ISO timestamp — only fetch runs created after this.
    per_page : Number of runs to fetch per repo.

    Returns
    -------
    list — List of workflow run dicts.
    """
    url = f"{GITHUB_API_BASE}/repos/{repo}/actions/runs"
    params = {"per_page": per_page, "status": "completed"}
    if since:
        params["created"] = f">{since}"

    try:
        response = await client.get(url, headers=HEADERS, params=params, timeout=15)
        if response.status_code == 200:
            return response.json().get("workflow_runs", [])
        elif response.status_code == 403:
            logger.warning("Rate limited for repo: %s", repo)
        elif response.status_code == 404:
            logger.warning("Repo not found: %s", repo)
        else:
            logger.warning("GitHub API error %d for %s", response.status_code, repo)
    except httpx.TimeoutException:
        logger.warning("Timeout fetching runs for: %s", repo)
    except Exception as exc:
        logger.error("Error fetching runs for %s: %s", repo, exc)

    return []


# ---------------------------------------------------------------------------
# Live prediction event generator
# ---------------------------------------------------------------------------

async def live_prediction_generator(
    predict_fn,
    poll_interval: int = POLL_INTERVAL_SECONDS,
) -> AsyncGenerator[str, None]:
    """
    Async generator that polls GitHub API and yields SSE events with
    live predictions for each new workflow run found.

    Parameters
    ----------
    predict_fn   : Callable that takes a feature dict and returns
                   {probability, prediction, risk_level, top_shap_features}.
    poll_interval: Seconds between GitHub API polls.

    Yields
    ------
    str — Server-Sent Event formatted string.
    """
    import json

    seen_run_ids: set = set()
    last_poll_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    logger.info("Live poller started — polling %d repos every %ds",
                len(MONITORED_REPOS), poll_interval)

    # Send initial connection event
    yield f"data: {json.dumps({'type': 'connected', 'message': 'PipelineGuard live stream started', 'repos': MONITORED_REPOS})}\n\n"

    async with httpx.AsyncClient() as client:
        while True:
            new_events = []

            for repo in MONITORED_REPOS:
                runs = await fetch_recent_runs(client, repo, since=None, per_page=5)

                for run in runs:
                    run_id = str(run.get("id", ""))
                    if run_id in seen_run_ids:
                        continue

                    seen_run_ids.add(run_id)

                    # Extract features
                    features = extract_features(run, repo)

                    # Get prediction
                    try:
                        result = predict_fn(features)
                    except Exception as exc:
                        logger.error("Prediction failed for run %s: %s", run_id, exc)
                        continue

                    event = {
                        "type": "prediction",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "run_id": run_id,
                        "repo": repo,
                        "pipeline_name": features["pipeline_name"],
                        "branch": features["head_branch"],
                        "trigger": features["trigger_type"],
                        "probability": result.get("probability", 0.0),
                        "prediction": result.get("prediction", 0),
                        "risk_level": result.get("risk_level", "LOW"),
                        "top_shap_features": result.get("top_shap_features", []),
                        "alert_sent": result.get("alert_sent", False),
                    }

                    new_events.append(event)
                    logger.info(
                        "New run | %s | %s | risk=%s | prob=%.3f",
                        repo, features["pipeline_name"],
                        event["risk_level"], event["probability"]
                    )

            # Yield all new events
            for event in new_events:
                yield f"data: {json.dumps(event)}\n\n"

            # Send heartbeat every poll cycle to keep connection alive
            heartbeat = {
                "type": "heartbeat",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "repos_polled": len(MONITORED_REPOS),
                "new_runs_found": len(new_events),
            }
            yield f"data: {json.dumps(heartbeat)}\n\n"

            await asyncio.sleep(poll_interval)