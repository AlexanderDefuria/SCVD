import argparse
from typing import List, Tuple

from pytorch_lightning import Trainer, seed_everything

from src.utils import download, interact
from src.preprocess import preprocess
from src.dataset import FunctionSourceCodeDataModule, GraphDataModule
from src.model import CodeBERTModel, GraphModel

if __name__ == "__main__":
    seed_everything(4343, workers=True)
    parser = argparse.ArgumentParser(description="Main script for the project.")
    parser.add_argument(
        "--project",
        type=str,
        choices=["ffmpeg", "qemu"],
        default="ffmpeg",
        help="Project to use for training/testing.",
    )
    parser.add_argument(  # TODO
        "--scope",
        type=str,
        choices=["all", "added", "modified", "deleted"],
        default="all",
        help="Scope of the project.",
    )
    parser.add_argument(  # TODO
        "--dataset",
        type=str,
        choices=["all", "cpg", "func_sc", "diff_sc"],
        default="all",
        help="Dataset to use for training/testing.",
    )

    project = parser.parse_args().project
    scope = parser.parse_args().scope

    download()
    commit_list: List[Tuple[str, str]] = preprocess(n=20)
    data_module = FunctionSourceCodeDataModule(project=project, batch_size=10)

    trainer = Trainer(
        fast_dev_run=False,
        accelerator="auto",
        devices=1,
        max_epochs=1,
    )
    trainer.fit(
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
