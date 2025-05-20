import argparse
from typing import List, Tuple

from pytorch_lightning import Trainer, seed_everything

from src.utils import download, interact
from src.preprocess import preprocess
from src.dataset import GraphDataModule
from src.model import GraphModel

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
    parser.add_argument(
        "--scope",
        type=str,
        choices=["all", "added", "modified", "deleted"],
        default="all",
        help="Scope of the project.",
    )

    project = parser.parse_args().project
    scope = parser.parse_args().scope

    download()
    commit_list: List[Tuple[str, str]] = preprocess(scope=scope, n=20)
    data_module = GraphDataModule(project=project, batch_size=10)
    data_module.prepare_data()
    data_module.setup(stage="fit")

    print("\n\nTrain")
    for batch in data_module.train_dataloader():
        print(batch)

    print("\n\nVal")
    for batch in data_module.val_dataloader():
        print(batch)

    print("\n\nTest")
    data_module.setup(stage="test")
    for batch in data_module.test_dataloader():
        print(batch)

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
