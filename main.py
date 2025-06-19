import argparse
from typing import List, Tuple

from pytorch_lightning import Trainer, seed_everything

from src.utils import download, interact
from src.preprocess import preprocess
from src.datasets.graph import GraphDataModule
from src.models.graph import GraphModel

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
    commit_list: List[str] = preprocess(n=200)
    data_module = GraphDataModule(project=project, batch_size=5, workers=11)
    data_module.setup()
    model = GraphModel(input_dim=4, mapping=data_module.abstraction_dictionary)

    trainer = Trainer(
        devices=1,
        min_epochs=5,
        max_epochs=10,
        log_every_n_steps=2,
        accelerator="gpu",
        strategy="ddp_spawn",
    )
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)

