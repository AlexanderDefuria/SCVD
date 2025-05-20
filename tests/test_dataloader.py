from pathlib import Path

from networkx import DiGraph, disjoint_union
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from src.dataset import data_path, get_statements
from src.preprocess import checkout_commit, get_commit_file_cpgs, get_commit_functions_sc, get_commit_lines_sc, get_data
from src.utils import download, interact

import pytest
import os


class TestDataLoader:
    # Before all
    def setup_method(self):
        download()
        assert os.path.exists(data_path()), "Data directory does not exist"

    def test_ffmpeg_download(self):
        project = "ffmpeg"
        assert os.path.exists(data_path() / "repo" / project), "Data directory does not exist"
        assert os.path.exists(data_path() / "repo" / project / "fftools/ffmpeg.c"), "ffmpeg.c directory does not exist"

    def test_qemu_download(self):
        project = "qemu"
        assert os.path.exists(data_path() / "repo" / project), "Data directory does not exist"
        assert os.path.exists(data_path() / "repo" / project / "job.c"), "qemu job.c directory does not exist"

    def test_function_sc_extraction(self):
        project = "ffmpeg"
        project_dir = Path(__file__).parent.parent / "data" / "repo" / project
        commit = "20502ba92a5936bc6a6e006d05828b750f4290ed"
        checkout_commit(commit, project_dir)

        # Get original function
        original_file_dir = project_dir / "fftools" / "ffmpeg.c"
        original_function = ""
        with open(original_file_dir, "r") as f:
            lines = f.readlines()
            original_function = "".join(lines[309:362])
            # https://github.com/FFmpeg/FFmpeg/commit/20502ba92a5936bc6a6e006d05828b750f4290ed#diff-8ec3b8f79e6d01d61c4ab09412160f29419415e60215431b23aaf5fec09bc9d1L311
            # To
            # https://github.com/FFmpeg/FFmpeg/commit/20502ba92a5936bc6a6e006d05828b750f4290ed#diff-8ec3b8f79e6d01d61c4ab09412160f29419415e60215431b23aaf5fec09bc9d1L362

        # Get function from commit
        functions = get_commit_functions_sc(commit, project_dir)
        extracted_function = "\n".join(functions[0]["lines_of_code"]) + "\n"

        assert original_function == extracted_function, "Function extraction failed"

    def test_diff_sc_extraction(self):
        project = "ffmpeg"
        project_dir = Path(__file__).parent.parent / "data" / "repo" / project
        commit = "20502ba92a5936bc6a6e006d05828b750f4290ed"
        checkout_commit(commit, project_dir)
        diff = get_commit_lines_sc(commit, project_dir)

        assert ("".join(diff[0]["lines_of_code"])).strip() == "if (print_graphs || print_graphs_file)"

    def test_graph_extraction(self): 
        project = "ffmpeg"
        project_dir = Path(__file__).parent.parent / "data" / "repo" / project
        commit = "20502ba92a5936bc6a6e006d05828b750f4290ed"
        checkout_commit(commit, project_dir)
        # TODO
