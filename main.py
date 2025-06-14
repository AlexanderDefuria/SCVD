import argparse
from pathlib import Path
from typing import List, Tuple

from pytorch_lightning import Trainer, seed_everything

from src import EXAMPLE_SC
from src.utils import download, interact
from src.preprocess import preprocess
from src.dataset import FunctionSourceCodeDataModule, GraphDataModule
from src.model import CodeBERTModel, GraphModel
from src.graphml import NodeId, Abstraction, Mappings, run_graphml

if __name__ == "__main__":
    seed_everything(1, workers=True)
    parser = argparse.ArgumentParser(description="Main script for the project.")
    parser.add_argument(
        "--project",
        type=str,
        choices=["ffmpeg", "qemu"],
        default="ffmpeg",
        help="Project to use for training/testing.",
    )
    parser.add_argument(  # TODO implement this.
        "--dataset",
        type=str,
        choices=["all", "cpg", "func_sc", "diff_sc"],
        default="all",
        help="Dataset to use for training/testing.",
    )

    project = parser.parse_args().project

    download()
    commit_list: List[Tuple[str, str]] = preprocess(n=10)
    data_module = GraphDataModule(project=project, batch_size=10)
    # data_module = FunctionSourceCodeDataModule(project=project, batch_size=10)
    exit()

    trainer = Trainer(
        fast_dev_run=False,
        devices=1,
        max_epochs=20,
        log_every_n_steps=1,
    )
    trainer.fit(
        model=CodeBERTModel(),
        datamodule=data_module,
    )
    trainer.test(
        model=CodeBERTModel(),
        datamodule=data_module,
    )

    # TODO
    #  BATCHING ISSUE
    #  When batching all `Data` objects are concatenated into a single disconnected graph
    #  That can then be passed to the model. This is similar to the row-wise batching process.
    #
    #  Issue arrises when not all graphs have the same properties
    #  Some don't have 'src' because the CFG might be a single line or two
    #   Ex.
    #       Data(edge_index=[2, 0], node=[1], name='AC3_WINDOW_SIZE', num_nodes=1, statements=[2], label='0')
    #     vs.
    #       Data(edge_index=[2, 44], node=[30], src=[44], dst=[44], name='s337m_probe', num_nodes=30, statements=[85], label='0')
