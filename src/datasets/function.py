from torch_geometric.data import (
    Data,
    Dataset as TorchGraphDataset,
)
from transformers.models.auto.tokenization_auto import AutoTokenizer
from torch.utils.data import DataLoader, Dataset as TorchDataset, random_split
from pytorch_lightning import LightningDataModule
from torch_geometric.data.data import BaseData
from typing import Any, Dict, List, Optional, Set
from src import VERBOSE
from src.datasets import VulnerabilityDataset, data_path, get_commits_from_cpg
import pickle
import torch
import os


from src.preprocess import get_data
from src.utils import interact


class FunctionSourceCodeDataset(VulnerabilityDataset, TorchDataset):
    def __init__(self, root: str, commit_list: List[str], transform=None):
        super().__init__(root, commit_list, transform=transform)
        self.processed_dir = str(data_path() / "processed" / "commit_function_sc")
        self.root = root
        self.commit_list = commit_list

    def __len__(self) -> int:
        return len(self.commit_list)

    def __getitem__(self, idx) -> BaseData:
        commit = self.commit_list[idx]
        file = f"{self.processed_dir}/{commit}.pt"
        return torch.load(
            file,
            map_location="cpu",
            weights_only=False,
        )


class FunctionSourceCodeDataModule(LightningDataModule):
    def __init__(self, project: str, batch_size: int = 32):
        """
        Function Source Code Data Module for loading the dataset.
        """
        super().__init__()
        self.project = project
        self.batch_size = batch_size
        self.dataset = FunctionSourceCodeDataset(
            root=str(data_path() / "commit_function_sc"),
            commit_list=get_commits_from_cpg(project),
        )
        self.root = str(data_path() / "processed" / "commit_function_sc")
        os.makedirs(self.root, exist_ok=True)

    def prepare_data(self):
        checkpoint = "microsoft/codebert-base"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        for file in os.listdir(self.dataset.root):
            if file.endswith(".pkl") and not file.endswith("skip.pkl"):
                commit = file.split("_")[1].split(".")[0]
                label = int(file.split("_")[2].split(".")[0])
                data = pickle.load(open(os.path.join(self.dataset.root, file), "rb"))
                source_code = ""

                for file in data:
                    source_code += "\n"
                    source_code += f"// {file['file']}"
                    source_code += "\n"
                    source_code += "\n".join(file["lines_of_code"])

                tokenized = tokenizer(
                    source_code,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                # Save the data
                torch.save(
                    {
                        "input_ids": tokenized["input_ids"].squeeze(0),
                        "attention_mask": tokenized["attention_mask"],
                        "label": label,
                    },
                    os.path.join(self.root, f"{commit}.pt"),
                )

    def setup(self, stage=None):
        commit_list = os.listdir(self.root)
        commit_list = [file.split(".")[0] for file in commit_list if file.endswith(".pt")]
        dataset = FunctionSourceCodeDataset(
            root=self.root,
            commit_list=commit_list,
        )
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        if not hasattr(self, "train_data"):
            self.train_data, self.val_data, self.test_data = random_split(
                dataset,
                [train_size, val_size, test_size],
            )

    def train_dataloader(self):
        assert self.train_data is not None, "train_data is not set. Call setup() first."
        loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=11)
        return loader

    def val_dataloader(self):
        assert self.val_data is not None, "val_data is not set. Call setup() first."
        loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=11)
        return loader

    def test_dataloader(self):
        assert self.test_data is not None, "test_data is not set. Call setup() first."
        loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=11)
        return loader
