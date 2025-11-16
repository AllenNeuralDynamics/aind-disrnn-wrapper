"""Logging utilities for wandb and repository tracking."""

import subprocess
import logging
import os
from pathlib import Path
import importlib.util

logger = logging.getLogger(__name__)

REPOS_TO_TRACK = {
    "disentangled-rnns": "disentangled_rnns",
    "aind-disrnn-utils": "aind_disrnn_utils",
}


def get_repository_commits():
    """
    Get commit hashes of important repositories from the installed environment.
    
    Returns:
        dict: Dictionary mapping repository names to their commit hashes
    """
    commits = {}
    for repo_name, module_name in REPOS_TO_TRACK.items():
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None and spec.origin is not None:
                repo_path = Path(spec.origin).parent
            else:
                logger.warning(f"Could not find module {module_name} for {repo_name}")
                continue
            
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                commits[repo_name] = result.stdout.strip()
                logger.info(f"{repo_name} commit: {commits[repo_name]}")
        except Exception as e:
            logger.warning(f"Could not get commit for {repo_name}: {e}")
    
    return commits


def get_computation_id():
    """
    Get the Code Ocean computation ID from the environment.
    
    Returns:
        str: The computation ID if available, None otherwise
    """
    computation_id = os.environ.get("CO_COMPUTATION_ID")
    if computation_id:
        logger.info(f"Code Ocean Computation ID: {computation_id}")
        return computation_id
    else:
        logger.warning("CO_COMPUTATION_ID not found in environment")
        return None


if __name__ == "__main__":
    
    commits = get_repository_commits()
    print("Repository Commits:", commits)
    computation_id = get_computation_id()
    print("Computation ID:", computation_id)