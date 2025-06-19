from src.datasets import VulnerabilityDataset, get_commits_from_cpg
from torch_geometric.data import  Dataset as TorchGraphDataset
from torch.utils.data import DataLoader, Dataset as TorchDataset, random_split
from torch_geometric.data.data import BaseData
from typing import Any, Dict, List, Optional, Set
from src.datasets import VulnerabilityDataset, data_path
import torch



class CommitDiffDataset(VulnerabilityDataset, TorchDataset):
    def __init__(self, root: str, commit_list: List[str], transform=None):
        super().__init__(root, commit_list, transform=transform)
        self.project = root
        self.commit_list: List[str] = get_commits_from_cpg(self.project)
        self.processed_dir = str(data_path() / "processed" / "commit_diff_sc")
        self.root = root

    def __len__(self) -> int:
        return len(self.commit_list)

    def __getitem__(self, idx) -> BaseData:
        commit = self.commit_list[idx]
        return torch.load(
            f"{self.processed_dir}/{self.project}_{commit}.pt",
            map_location="cpu",
            weights_only=False,
        )



