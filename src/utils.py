from pathlib import Path
import torch
from torch_geometric.data import Data
import code
import pygit2 as git
import os


VERBOSE = " > /dev/null 2>&1"

def download():
    root = Path(__file__).parent.parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = data_dir / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    # Check if ffmpeg and qemu directories exist
    ffmpeg_dir = repo_dir / "ffmpeg"
    qemu_dir = repo_dir / "qemu"

    if not ffmpeg_dir.exists():
        print("Downloading ffmpeg repository...")
        os.system("git clone git@github.com:FFmpeg/FFmpeg.git " + str(ffmpeg_dir))

    if not qemu_dir.exists():
        print("Downloading qemu repository...")
        os.system("git clone git@github.com:qemu/qemu.git " + str(qemu_dir))


def data_dir():
    root = Path(__file__).parent.parent
    data_dir = root / "data"
    return data_dir


def interact(locals, end=True):
    code.interact(local=locals)
    if end:
        exit()


def combine_subgraphs(subgraph_list):
    # Initialize lists to store combined attributes
    all_x = []
    all_edge_index = []
    node_offset = 0

    for graph in subgraph_list:
        # Add node features
        all_x.append(graph.x)

        # Adjust edge indices and add them
        if hasattr(graph, "edge_index") and graph.edge_index is not None:
            adjusted_edge_index = graph.edge_index.clone()
            adjusted_edge_index += node_offset
            all_edge_index.append(adjusted_edge_index)

        # Update offset for the next graph
        node_offset += graph.x.size(0)

    # Combine everything into a single Data object
    combined_x = torch.cat(all_x, dim=0)
    combined_edge_index = torch.cat(all_edge_index, dim=1) if all_edge_index else None

    return Data(x=combined_x, edge_index=combined_edge_index)


def checkout_commit(commit: str, project_dir: Path, before: bool = True) -> None:
    """
    Checkout the commit using git
    git -C {project_dir} checkout {commit}
    NOTE: This checks out the predecessor commit to get the BEFORE by default
    """
    os.system(f"git -C {project_dir} checkout -f {commit}{'^' if before else ''} {VERBOSE}")
    # interact(locals())


def ignore_file(file: Path) -> bool:
    """
    Ignore the file if it is not a C/C++ file
    TRUE indicates we should ignore the file
    """
    # TODO Update the filter based on ICVul Extension Filter.
    filename = file.name
    if not (filename.endswith(".c") or filename.endswith(".cpp") or filename.endswith(".h")):
        if VERBOSE != " > /dev/null 2>&1":
            print(f"Ignoring file: {filename}")
        return True
    return False


def get_diff(commit: str, project_dir: Path) -> git.Diff:
    repo = git.init_repository(project_dir)
    diff = repo.diff(commit + "^", commit, context_lines=0, interhunk_lines=0)
    assert isinstance(diff, git.Diff)
    return diff
