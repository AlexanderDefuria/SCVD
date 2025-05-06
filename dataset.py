from typing import List
import torch 
from torch_geometric.data import Dataset
import os

from preprocess import get_data


class GraphDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.project = root
        self.commit_list: List[str] = get_data(self.project)['sha_id'].to_list()
        self.root = f"data/processed/{self.project}/"
        
        super(GraphDataset, self).__init__(self.root, transform=transform)

    @property 
    def processed_file_names(self):
        # Read from the processed directory
        return [f"{self.project}_{commit}.pt" for commit in self.commit_list]
        
    def len(self):
        return len(self.commit_list)

    def get(self, idx):
        commit = self.commit_list[idx]
        data = torch.load(os.path.join(self.processed_dir, f"{self.project}_{commit}.pt"))
        return data

# TODO Encoder

