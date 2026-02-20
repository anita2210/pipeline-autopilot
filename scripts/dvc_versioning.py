"""
dvc_versioning.py
-----------------
Automates DVC data versioning: dvc add, dvc push, and git commit of .dvc files.
Tracks raw and processed datasets alongside git commits.

Author  : Member 2 (Data Engineer)
Project : Pipeline Autopilot — CI/CD Failure Prediction System
Date    : February 2026
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.config import (
    DVC_CONFIG,
    LOGGING_CONFIG,
    PROCESSED_DATASET_PATH,
    RAW_DATASET_PATH,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["log_level"], logging.INFO),
    format=LOGGING_CONFIG["log_format"],
    datefmt=LOGGING_CONFIG["date_format"],
)
logger = logging.getLogger("dvc_versioning")

# ---------------------------------------------------------------------------
# Constants from config
# ---------------------------------------------------------------------------
DVC_REMOTE_NAME = DVC_CONFIG["remote_name"]   # "gcs_remote"
DVC_REMOTE_URL  = DVC_CONFIG["remote_url"]    # "gs://pipeline-autopilot-data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list, error_msg: str) -> subprocess.CompletedProcess:
    """
    Run a shell command and raise a clear RuntimeError on failure.

    Parameters
    ----------
    cmd       : list — Command + arguments.
    error_msg : str  — Human-readable context for the error.

    Returns
    -------
    subprocess.CompletedProcess
    """
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("%s\nSTDOUT: %s\nSTDERR: %s", error_msg, result.stdout, result.stderr)
        raise RuntimeError(f"{error_msg}\n{result.stderr.strip()}")
    if result.stdout.strip():
        logger.info(result.stdout.strip())
    return result


def _is_dvc_initialized() -> bool:
    """Check whether DVC has been initialized in the current repo."""
    return Path(".dvc").exists()


def _is_git_repo() -> bool:
    """Check whether the current directory is inside a git repo."""
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        capture_output=True, text=True
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Core DVC functions
# ---------------------------------------------------------------------------

def dvc_init() -> None:
    """
    Initialize DVC in the current repo (safe to call if already initialized).
    Sets up the GCS remote from DVC_CONFIG.
    """
    if _is_dvc_initialized():
        logger.info("DVC already initialized - skipping init.")
        return

    if not _is_git_repo():
        raise RuntimeError("Not a git repository. Run 'git init' first.")

    logger.info("Initializing DVC...")
    _run(["dvc", "init"], "DVC initialization failed.")

    logger.info("Configuring remote: %s -> %s", DVC_REMOTE_NAME, DVC_REMOTE_URL)
    _run(
        ["dvc", "remote", "add", "-d", DVC_REMOTE_NAME, DVC_REMOTE_URL],
        "Failed to add DVC remote."
    )
    logger.info("DVC initialized and remote configured.")


def dvc_add(file_path: Path) -> Path:
    """
    Add a data file to DVC tracking (creates/updates the .dvc pointer file).

    Parameters
    ----------
    file_path : Path — Path to the data file to track.

    Returns
    -------
    Path — Path to the generated .dvc file.

    Raises
    ------
    FileNotFoundError : If the data file does not exist.
    RuntimeError      : If DVC is not initialized.
    """
    if not _is_dvc_initialized():
        raise RuntimeError("DVC not initialized. Run dvc_init() first.")

    if not file_path.exists():
        raise FileNotFoundError(f"Cannot track non-existent file: {file_path}")

    logger.info("Running dvc add on: %s", file_path)
    _run(["dvc", "add", str(file_path)], f"dvc add failed for {file_path}")

    dvc_file = Path(str(file_path) + ".dvc")
    logger.info("DVC pointer created: %s", dvc_file)
    return dvc_file


def dvc_push(remote: Optional[str] = None) -> None:
    """
    Push all tracked data files to the DVC remote storage.

    Parameters
    ----------
    remote : str, optional — Remote name. Defaults to DVC_REMOTE_NAME from config.

    Raises
    ------
    RuntimeError : If push fails or remote is unreachable.
    """
    if not _is_dvc_initialized():
        raise RuntimeError("DVC not initialized. Run dvc_init() first.")

    target_remote = remote or DVC_REMOTE_NAME
    logger.info("Pushing data to DVC remote: %s", target_remote)

    _run(
        ["dvc", "push", "--remote", target_remote],
        f"dvc push to '{target_remote}' failed. Check remote connectivity."
    )
    logger.info("dvc push complete.")


def git_commit_dvc_files(message: str = "chore: update DVC tracked data files") -> None:
    """
    Stage all .dvc files and .gitignore updates, then commit to git.

    Parameters
    ----------
    message : str — Git commit message.
    """
    if not _is_git_repo():
        raise RuntimeError("Not a git repo. Cannot commit.")

    logger.info("Staging .dvc files for git commit...")

    _run(
        ["git", "add", "*.dvc", "**/*.dvc", "**/.gitignore", ".gitignore"],
        "git add of .dvc files failed."
    )

    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        capture_output=True
    )
    if status.returncode == 0:
        logger.info("Nothing new to commit - .dvc files unchanged.")
        return

    _run(["git", "commit", "-m", message], "git commit failed.")
    logger.info("Git commit created: '%s'", message)


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def version_raw_data(commit: bool = True) -> None:
    """
    Track the raw dataset with DVC and optionally commit the .dvc file.

    Parameters
    ----------
    commit : bool — Whether to auto-commit the .dvc file to git.
    """
    logger.info("--- Versioning RAW data ---")
    dvc_add(RAW_DATASET_PATH)
    if commit:
        git_commit_dvc_files("chore: track raw dataset with DVC")


def version_processed_data(commit: bool = True) -> None:
    """
    Track the processed dataset with DVC and optionally commit the .dvc file.

    Parameters
    ----------
    commit : bool — Whether to auto-commit the .dvc file to git.
    """
    logger.info("--- Versioning PROCESSED data ---")
    if not PROCESSED_DATASET_PATH.exists():
        logger.warning(
            "Processed dataset not found at %s - skipping.", PROCESSED_DATASET_PATH
        )
        return
    dvc_add(PROCESSED_DATASET_PATH)
    if commit:
        git_commit_dvc_files("chore: track processed dataset with DVC")


def run_full_versioning(push: bool = True, commit: bool = True) -> None:
    """
    Full DVC versioning pipeline: add raw + processed files, push to remote.

    Parameters
    ----------
    push   : bool — Whether to push to remote after tracking.
    commit : bool — Whether to auto git-commit .dvc files.
    """
    logger.info("=" * 60)
    logger.info("DVC VERSIONING START")
    logger.info("=" * 60)

    if not _is_dvc_initialized():
        logger.info("DVC not found - initializing first...")
        dvc_init()

    version_raw_data(commit=commit)
    version_processed_data(commit=commit)

    if push:
        dvc_push()

    logger.info("DVC VERSIONING COMPLETE")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline Autopilot - DVC Versioning"
    )
    parser.add_argument(
        "--no-push", action="store_true",
        help="Skip dvc push (useful for local testing)"
    )
    parser.add_argument(
        "--no-commit", action="store_true",
        help="Skip git commit of .dvc files"
    )
    args = parser.parse_args()

    run_full_versioning(
        push=not args.no_push,
        commit=not args.no_commit,
    )