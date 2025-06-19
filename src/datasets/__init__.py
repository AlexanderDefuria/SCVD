from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import abc
import os


def data_path() -> Path:
    return Path(__file__).parent.parent.resolve() / "data"


def get_commits_from_cpg(project: str) -> List[str]:
    """
    Get the commits from the cpg directory.
    """
    commits = []
    for file in os.listdir(data_path() / "commit_file_sc"):
        if file.endswith(".pkl") and file.startswith(project) and "skip" not in file:
            commit = file.split("_")[1]
            commits.append(commit)
    return commits


class VulnerabilityDataset(abc.ABC):
    def __init__(self, root: str, commit_list: List[str], transform=None):
        self.project = root
        self.commit_list: List[str] = get_commits_from_cpg(self.project)
        self.root = root


