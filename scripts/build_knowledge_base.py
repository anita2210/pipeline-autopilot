"""
build_knowledge_base.py
-----------------------
Precomputes knowledge base from final_dataset_processed.csv and saves as 5 files:
- global_stats.json
- daily_stats.json
- repo_stats.json
- error_stats.json
- similar_runs_index.pkl (FAISS index)
"""

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

DATA_PATH = Path("scripts/final_dataset_processed.csv")
KB_DIR = Path("knowledge_base")
KB_DIR.mkdir(exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Loaded {len(df)} rows")

"""
build_knowledge_base.py
-----------------------
Precomputes knowledge base from final_dataset_processed.csv and saves as 5 files:
- global_stats.json
- daily_stats.json
- repo_stats.json
- error_stats.json
- similar_runs_index.pkl (FAISS index)
"""

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

DATA_PATH = Path("scripts/final_dataset_processed.csv")
KB_DIR = Path("knowledge_base")
KB_DIR.mkdir(exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Loaded {len(df)} rows")

# 1. global_stats.json
print("Building global_stats.json...")
global_stats = {
    "total_runs": int(len(df)),
    "total_failures": int(df["failed"].sum()),
    "failure_rate": round(float(df["failed"].mean()), 4),
    "avg_duration_seconds": round(float(df["duration_seconds"].mean()), 2),
    "avg_retry_count": round(float(df["retry_count"].mean()), 4),
    "avg_failures_last_7_runs": round(float(df["failures_last_7_runs"].mean()), 4),
    "avg_workflow_failure_rate": round(float(df["workflow_failure_rate"].mean()), 4),
    "top_failure_types": df[df["failed"]==1]["failure_type"].value_counts().head(5).to_dict(),
    "top_error_messages": df[df["failed"]==1]["error_message"].value_counts().head(5).to_dict(),
}
with open(KB_DIR / "global_stats.json", "w") as f:
    json.dump(global_stats, f, indent=2)
print("  global_stats.json saved")

# 2. daily_stats.json
print("Building daily_stats.json...")
daily = {}
for day in sorted(df["day_of_week"].unique()):
    sub = df[df["day_of_week"] == day]
    daily[str(int(day))] = {
        "failure_rate": round(float(sub["failed"].mean()), 4),
        "total_runs": int(len(sub)),
        "total_failures": int(sub["failed"].sum()),
    }
hourly = {}
for hour in sorted(df["hour"].unique()):
    sub = df[df["hour"] == hour]
    hourly[str(int(hour))] = {
        "failure_rate": round(float(sub["failed"].mean()), 4),
        "total_runs": int(len(sub)),
    }
daily_stats = {"by_day_of_week": daily, "by_hour": hourly}
with open(KB_DIR / "daily_stats.json", "w") as f:
    json.dump(daily_stats, f, indent=2)
print("  daily_stats.json saved")

# 3. repo_stats.json
print("Building repo_stats.json...")
repo_stats = {}
for repo in df["repo"].value_counts().head(20).index:
    sub = df[df["repo"] == repo]
    repo_stats[str(repo)] = {
        "failure_rate": round(float(sub["failed"].mean()), 4),
        "total_runs": int(len(sub)),
        "total_failures": int(sub["failed"].sum()),
        "avg_duration": round(float(sub["duration_seconds"].mean()), 2),
    }
with open(KB_DIR / "repo_stats.json", "w") as f:
    json.dump(repo_stats, f, indent=2)
print("  repo_stats.json saved")

# 4. error_stats.json
print("Building error_stats.json...")
failed_df = df[df["failed"] == 1]
error_stats = {}
for error in failed_df["error_message"].value_counts().head(20).index:
    sub = failed_df[failed_df["error_message"] == error]
    error_stats[str(error)] = {
        "count": int(len(sub)),
        "percentage": round(float(len(sub) / len(failed_df)), 4),
        "suggested_fix": "Check logs and retry. Review recent code changes.",
    }
with open(KB_DIR / "error_stats.json", "w") as f:
    json.dump(error_stats, f, indent=2)
print("  error_stats.json saved")

# 5. similar_runs_index.pkl (FAISS)
print("Building FAISS index...")
FEATURE_COLS = [
    "duration_seconds", "avg_duration_7runs" if "avg_duration_7runs" in df.columns else "avg_duration_7_runs",
    "duration_deviation", "failures_last_7_runs",
    "workflow_failure_rate", "retry_count", "concurrent_runs",
    "total_jobs", "failed_jobs", "hour", "day_of_week",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
features_df = df[FEATURE_COLS].fillna(0).astype(float)
vectors = features_df.values.astype(np.float32)

# Normalize
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
norms[norms == 0] = 1
vectors_norm = vectors / norms

index = faiss.IndexFlatL2(vectors_norm.shape[1])
index.add(vectors_norm)

faiss_data = {
    "index": index,
    "feature_cols": FEATURE_COLS,
    "labels": df["failed"].values.tolist(),
    "failure_types": df["failure_type"].values.tolist(),
}
with open(KB_DIR / "similar_runs_index.pkl", "wb") as f:
    pickle.dump(faiss_data, f)
print("  similar_runs_index.pkl saved")

print("\nKnowledge base built successfully!")
print(f"Files in {KB_DIR}:")
for f in KB_DIR.iterdir():
    print(f"  {f.name}")
