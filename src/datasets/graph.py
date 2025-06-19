from torch_geometric.data import Dataset as TorchGraphDataset
from torch_geometric.loader import DataLoader as TorchGraphDataLoader
from torch.utils.data import  random_split
from pytorch_lightning import LightningDataModule
from torch_geometric.data.data import BaseData
from networkx import DiGraph, MultiDiGraph
from typing import Any, Dict, List, Optional, Set
from src import VERBOSE
from src.datasets import VulnerabilityDataset, data_path, get_commits_from_cpg
from src.graphml import (
    Mappings,
    build_abstraction_mapping,
    cfg_to_pyg_data,
    extract_assignments,
    load_assignment_cfg,
    parse_joern_cpg,
    save_cfg_assignments,
    write_code_to_temp_file,
)
from tqdm import tqdm
import multiprocessing
import polars as pl
import pickle
import torch
import glob
import os


from src.preprocess import get_data
from src.utils import interact


def create_abstraction_dictionary(train_data, k: Optional[int] = None) -> Mappings:
    raw_occurrences: Dict[str, List] = {
        "data_type": [],
        "api_call": [],
        "constant": [],
        "operator": [],
    }
    for data in train_data:
        raw_occurrences['data_type'].extend(data.data_type)
        raw_occurrences['api_call'].extend(data.api_call)
        raw_occurrences['constant'].extend(data.constant)
        raw_occurrences['operator'].extend(data.operator)
    mapping: Mappings = build_abstraction_mapping(abstraction_dict=raw_occurrences)
    return mapping





class GraphDataset(VulnerabilityDataset, TorchGraphDataset):
    """
    Graph Dataset for loading the commit graphs. Note this is not for the RL portion of this project,
    simply for the graph-based vulnerability detection baseline without local context retreival.
    """

    def __init__(self, root: str, commit_list: List[str], transform=None):
        VulnerabilityDataset.__init__(self, root, commit_list, transform=transform)
        TorchGraphDataset.__init__(self, root, transform=transform)

    @property
    def processed_dir(self) -> str:
        # Processed data is stored in the processed directory
        return str(data_path() / "processed" / "graph")

    @property
    def processed_file_names(self):
        # rename = lambda x: x.replace(".pkl", ".pt")
        # return list(map(rename, self.raw_file_names))

        # Read from the processed directory
        files = glob.glob(os.path.join(self.processed_dir, f"{self.project}_*.pt"))
        return files

    @property
    def raw_dir(self) -> str:
        return str(data_path() / "commit_file_sc")

    @property
    def raw_paths(self) -> List[str]:
        """The absolute raw paths that must be present to skip downloading."""
        return [
            str(data_path() / "commit_file_sc" / f"{self.project}_{commit}.pkl")
            for commit in get_data(self.project)["sha_id"].to_list()
            # The Raw commit list. Note some of these are skipped.
        ]

    @property
    def raw_file_names(self):
        # Raw data is stored in the raw directory
        # return [f"{self.project}_{commit}.pt" for commit in self.commit_list]
        raw_files = glob.glob(os.path.join(self.raw_dir, f"{self.project}_*_*.pkl"))
        raw_files = [file for file in raw_files if "skip" not in file]
        return raw_files

    def process(self):
        """
        Write to disk for joern, parse CPG, get assignments, build abstraction dict.
        Abstractions are actually applied in the dataloader
        """
        print(f"Processing {self.project} dataset")
        os.makedirs(self.processed_dir, exist_ok=True)
        pickle_files = self.raw_file_names
        df = pl.DataFrame()
        for file in tqdm(pickle_files, desc="Loading commits", disable=VERBOSE != ""):
            if "skip" in file:
                continue
            file_df = pl.DataFrame(
                pickle.load(open(file, "rb"))["data"],
                schema=[("commit", str), ("file", str), ("lines_of_code", str)],
            )
            # Add label column
            label = file.split("_")[-1].split(".")[0]
            file_df = file_df.with_columns(
                pl.lit(label).alias("label"),
            )
            df.vstack(
                file_df,
                in_place=True,
            )
            df.rechunk()

        for row in tqdm(df.iter_rows(named=True), desc="Processing commits", disable=VERBOSE != ""):
            commit = row["commit"]
            file = row["file"]
            lines_of_code = row["lines_of_code"]
            label = row["label"]
            data = {file: lines_of_code}
            project_commit = f"{self.project}_{commit}"
            graph_name = f"{project_commit}.graphml"

            temp_file = write_code_to_temp_file(data, project_commit)
            cpg: MultiDiGraph = parse_joern_cpg(temp_file, project_commit)
            cfg: DiGraph = extract_assignments(cpg)
            save_cfg_assignments(graph_name, cfg)
            cfg = load_assignment_cfg(graph_name)
            pyg_data: BaseData = cfg_to_pyg_data(cfg)
            pyg_data.label = torch.tensor(int(label), dtype=torch.long)
            abstraction_values: List[Dict] = pyg_data.ABSTRACTION
            pyg_data.constant = [assignment.get("constant", None) for assignment in abstraction_values]
            pyg_data.operator = [assignment.get("operator", None) for assignment in abstraction_values]
            pyg_data.api_call = [assignment.get("api_call", None) for assignment in abstraction_values]
            pyg_data.data_type = [assignment.get("data_type", None) for assignment in abstraction_values]
            del pyg_data.ABSTRACTION  # Remove the original abstraction list
            del pyg_data.labelE  # Remove the label from the graph data

            torch.save(pyg_data, f"{self.processed_dir}/{self.project}_{commit}.pt")
            print(f"Saved Graph {self.project}_{commit}.pt")

        print(f"Loaded {len(df)} commits from {self.project} dataset in graph")

    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx) -> BaseData:
        commit = self.commit_list[idx]
        data = torch.load(
            f"{self.processed_dir}/{self.project}_{commit}.pt",
            map_location="cuda",
            weights_only=False,
        )
        return data


class GraphDataModule(LightningDataModule):
    def __init__(self, project: str, batch_size: int = 32, workers: Optional[int] = None):
        """
        Graph Data Module for loading the dataset.

        Args:
            project: str
            batch_size: int = 32
        """
        super().__init__()
        self.project = project
        self.batch_size = batch_size
        self.workers = multiprocessing.cpu_count() if not workers else workers
        if self.workers > 1:
            self.workers-=1

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        dataset = GraphDataset(root=self.project, commit_list=get_commits_from_cpg(self.project))
        total_size = len(dataset)
        train_size = int(0.4 * total_size)
        val_size = int(0.5 * total_size)
        test_size = total_size - train_size - val_size

        if not hasattr(self, "train_data") or not hasattr(self, "abstraction_dictionary"):
            self.train_data, self.val_data, self.test_data = random_split(
                dataset,
                [train_size, val_size, test_size],
            )
            self.abstraction_dictionary = create_abstraction_dictionary(self.train_data)

    def train_dataloader(self):
        assert self.train_data is not None, "train_data is not set. Call setup() first."
        return TorchGraphDataLoader(
            self.train_data,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        assert self.val_data is not None, "val_data is not set. Call setup() first."
        return TorchGraphDataLoader(
            self.val_data,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        assert self.test_data is not None, "test_data is not set. Call setup() first."
        return TorchGraphDataLoader(
            self.test_data,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.workers,
            persistent_workers=True,
        )


