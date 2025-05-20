from networkx.algorithms import union
from networkx.algorithms.operators import disjoint_union
from pyjoern.cfg.jil.statement import Statement
from torch_geometric.data import (
    Data,
    Dataset as TorchGraphDataset,
)
from transformers.models.auto.tokenization_auto import AutoTokenizer
from torch_geometric.loader import DataLoader as TorchGraphDataLoader
from torch.utils.data import DataLoader, Dataset as TorchDataset, random_split
from pytorch_lightning import LightningDataModule
from torch_geometric.data.data import BaseData
from networkx import DiGraph
from pathlib import Path
from torch import Tensor
from typing import List
import pickle
import torch
import glob
import abc
import os

from torch_geometric.utils import from_networkx

from src.preprocess import get_data
from src.utils import interact


def data_path() -> Path:
    return Path(__file__).parent.parent.resolve() / "data"


def get_commits_from_cpg(project: str) -> List[str]:
    """
    Get the commits from the cpg directory.
    """
    commits = []
    for file in os.listdir(data_path() / "commit_cpgs"):
        if file.endswith(".pkl") and file.startswith(project) and "skip" not in file:
            commit = file.split("_")[1]
            commits.append(commit)
    return commits


def get_statements(graph: Data) -> List[Statement]:
    """
    Get the statements from the graph
    """
    statements = []
    try:
        for node in graph.node:
            if str(node.__class__) == "<class 'pyjoern.cfg.jil.block.Block'>":
                statements.extend(node.statements)
        return statements
    except Exception as e:
        print(f"Error getting statements: {e}")
        return []


class VulnerabilityDataset(abc.ABC):
    def __init__(self, root: str, commit_list: List[str], transform=None):
        self.project = root
        self.commit_list: List[str] = get_commits_from_cpg(self.project)
        self.root = root


class GraphDataset(VulnerabilityDataset, TorchGraphDataset):
    def __init__(self, root: str, commit_list: List[str], transform=None):
        VulnerabilityDataset.__init__(self, root, commit_list, transform=transform)
        TorchGraphDataset.__init__(self, root, transform=transform)

    @property
    def processed_dir(self) -> str:
        # Processed data is stored in the processed directory
        return str(data_path() / "processed" / "graph")

    @property
    def processed_file_names(self):
        # Read from the processed directory
        rename = lambda x: x.replace(".pkl", ".pt")
        return list(map(rename, self.raw_file_names))

    @property
    def raw_dir(self) -> str:
        return str(data_path() / "commit_cpgs")

    @property
    def raw_paths(self) -> List[str]:
        """The absolute raw paths that must be present to skip downloading."""
        return [
            str(data_path() / "commit_cpgs" / f"{self.project}_{commit}.pkl")
            for commit in get_data(self.project)["sha_id"].to_list()
            # The Raw commit list. Note some of these are skipped.
        ]

    @property
    def raw_file_names(self):
        # Raw data is stored in the raw directory
        # return [f"{self.project}_{commit}.pt" for commit in self.commit_list]
        return glob.glob(os.path.join(self.raw_dir, f"{self.project}_*_*.pkl"))

    def process(self):
        print(f"Processing {self.project} dataset")
        for file in self.raw_file_names:
            # Load the altered function CFG's from the pickle file
            commit = file.split("/")[-1].split("_")[1].split(".")[0]
            label = file.split("/")[-1].split("_")[2].split(".")[0]
            if label == "skip":
                continue

            data = pickle.load(open(file, "rb"))

            cfg = DiGraph()
            names = []
            for cpg in data:
                names.append(cpg.cfg.name)  # TODO CONFIRM THIS WORKS.
                cfg = disjoint_union(cfg, cpg.cfg)

            graph: Data = from_networkx(cfg)
            statements = get_statements(graph)
            graph.statements = statements
            graph.label = label
            graph.commit = commit
            graph.name = names

            # TODO HACK For missing SRC and DST
            if not hasattr(graph, "src"):
                graph.src = names
            if not hasattr(graph, "dst"):
                graph.dst = names

            # Save the graph to the processed directory
            torch.save(graph, f"{self.processed_dir}/{self.project}_{commit}.pt")

    def map_into_dictionary(self):
        pass

    def len(self) -> int:
        return len(self.commit_list)

    def get(self, idx) -> BaseData:
        commit = self.commit_list[idx]
        return torch.load(
            f"{self.processed_dir}/{self.project}_{commit}.pt",
            map_location="cpu",
            weights_only=False,
        )


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


def create_abstraction_dictionary(train_data):
    pass


def get_data_loader(project: str, commit_list: List[str], batch_size: int, shuffle: bool, train: bool):
    dataset = GraphDataset(root=project, commit_list=commit_list)
    data_loader = TorchGraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
    )
    # TODO figure out how to use the train/test split
    # TODO REMOVE IN FAVOUR OF DATA MODULE

    return data_loader


class GraphDataModule(LightningDataModule):
    def __init__(self, project: str, batch_size: int = 32):
        """
        Graph Data Module for loading the dataset.

        Args:
            project: str
            batch_size: int = 32
        """
        super().__init__()
        self.project = project
        self.batch_size = batch_size
        self.dataset = GraphDataset(root=project, commit_list=get_commits_from_cpg(project))

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        dataset = GraphDataset(root=self.project, commit_list=get_commits_from_cpg(self.project))
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        if not hasattr(self, "train_data"):
            self.train_data, self.val_data, self.test_data = random_split(
                dataset,
                [train_size, val_size, test_size],
            )

        if stage == "fit" or stage is None:
            self.train_data = self.train_data
            self.val_data = self.val_data
            self.abstraction_dictionary = create_abstraction_dictionary(self.train_data)

        if stage == "test" or stage is None:
            self.test_data = self.test_data
            pass

    def train_dataloader(self):
        assert self.train_data is not None, "train_data is not set. Call setup() first."
        return TorchGraphDataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)  # type: ignore

    def val_dataloader(self):
        return TorchGraphDataLoader(self.val_data, batch_size=self.batch_size)  # type: ignore

    def test_dataloader(self):
        return TorchGraphDataLoader(self.test_data, batch_size=self.batch_size)  # type: ignore


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
        loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=1)
        return loader
