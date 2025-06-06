from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import pygit2 as git
import polars as pl
from pyjoern import parse_source # used to collect functions and their source code, NOT CPG's
from src.utils import interact, ignore_file, get_diff, checkout_commit
from networkx.drawing.nx_pydot import write_dot
import pickle
import os

COMMIT_FILE_SC = Path("data/commit_file_sc")
COMMIT_DIFF_SC = Path("data/commit_diff_sc")
COMMIT_FUNCTION_SC = Path("data/commit_function_sc")
TEST = True


def get_commit_scope(diff: git.Diff, project_name) -> Tuple[set, dict]:
    # Get list of files changed in the commit
    # Filter non C/C++ files
    # Get CPG of project at commit
    # Query lines in files changed to find functions in Joern.
    #     - Query function by METHOD.LINE_NUMBER and METHOD.LINE_NUMBER_END
    #     - Create set of METHODs for each file
    #     - What to do with class?
    # Collect CPG for selected Methods
    files = set()
    lines = {}

    # Grab the files that are changed in the commit
    # Filter out non C/C++ files so we can parse everything.
    for obj in diff.deltas:  # type: ignore
        obj: git.DiffDelta = obj
        file = Path(__file__).parent.parent / "data" / "repo" / project_name / obj.old_file.path
        if not ignore_file(file) and file.exists():
            files.add(obj.old_file.path)
            if file not in lines:
                lines[obj.old_file.path] = set()

    # Iterate over the changes
    # Get the lines that are changed in the commit
    for patch in diff:  # type: ignore
        if patch.delta.old_file.path not in lines:
            continue
        for hunk in patch.hunks:
            for line in hunk.lines:
                if line.old_lineno != -1 and line.old_lineno != 0:
                    lines[patch.delta.old_file.path].add(line.old_lineno)

    # List of file names relative to project, not absolute paths
    # List of lines that are changed in the commit
    return files, lines


def get_commit_file_sc(commit: str, project_dir: Path) -> Dict[str, List[Dict] | str]:
    """
    Get the files that are added/modified/deleted in the commit
    """
    project_name = project_dir.name
    diff = get_diff(commit, project_dir)
    files, lines = get_commit_scope(diff, project_name)
    out_files = []
    for file in files:
        with open(project_dir / file, "r") as f:
            out_files.append(
                {
                    "file": file,
                    "commit": commit,
                    "lines_of_code": f.read(),
                }
            )

    return {
        "type": "commit_file_sc",
        "data": out_files,
    }


def get_commit_functions_sc(commit: str, project_dir: Path) -> Dict[str, List[Dict] | str]:
    """
    Get the functions that are added/modified/deleted in the commit
    This only returns the source code of the functions that have been changed in the commit. (Including unchanged lines within those functions)
    """

    project_name = project_dir.name
    diff = get_diff(commit, project_dir)
    files, lines = get_commit_scope(diff, project_name)
    functions = []

    for file in files:
        # Get the CPG for the file
        # Get the functions in the file
        cfg = parse_source(project_dir / file)
        for function_name, function in cfg.items():
            source_code: str = ""
            if function.start_line is None or function.end_line is None:
                continue
            # Check if the function has changed lines, add if it does
            if not lines[file].isdisjoint(set(range(function.start_line, function.end_line + 1))):
                file_path = Path(__file__).parent.parent / "data" / "repo" / project_name / file
                with open(file_path, "r") as f:
                    source_code = f.read()
                    # Get the lines of code in the function
                    lines_of_code = source_code.splitlines()[function.start_line - 1 : function.end_line]
                    functions.append(
                        {
                            "name": function_name,
                            "commit": commit,
                            "file": file,
                            "start_line": function.start_line,
                            "end_line": function.end_line,
                            "lines_of_code": lines_of_code,
                        }
                    )
    return {
        "type": "commit_function_sc",
        "data": functions,
    }


def get_commit_lines_sc(commit: str, project_dir: Path) -> Dict[str, List[Dict] | str]:
    project_name = project_dir.name
    diff = get_diff(commit, project_dir)
    files, lines = get_commit_scope(diff, project_name)
    commits = []

    # Get the changed lines from the old file.
    for file in files:
        line_numbers = lines[file]
        source_code = ""
        with open(project_dir / file, "r") as f:
            source_code = f.read()
            # Get the lines of code in the function
            lines_of_code = source_code.splitlines()
            # Get the lines that are changed in the commit
            lines[file] = [lines_of_code[line - 1] for line in line_numbers]

        commits.append(
            {
                "file": file,
                "commit": commit,
                "lines_of_code": lines[file],
            }
        )

    return {
        "type": "commit_diff_sc",
        "data": commits,
    }


def save_target(pickle_file: Path, target: Dict[str, List[Dict] | str]) -> None:
    # Pickle the cpgs to disk
    pickle.dump(target, open(pickle_file, "wb"))

    # Compare the cpgs to the pickle file
    if TEST:
        reloaded_cpgs = pickle.load(open(pickle_file, "rb"))
        assert len(target) == len(reloaded_cpgs), "Saved Target does not match loaded target"
        # TODO a proper comparison of the cpgs for content


def validate_structure() -> None:
    assert Path("data").exists(), "data directory does not exist"
    assert Path("data/repo").exists(), "repo directory does not exist"
    assert Path("data/repo/ffmpeg").exists(), "ffmpeg directory does not exist"
    assert Path("data/repo/qemu").exists(), "qemu directory does not exist"
    assert Path("data/ffmpeg.csv").exists(), "ffmpeg.csv file does not exist"
    assert Path("data/qemu.csv").exists(), "qemu.csv file does not exist"
    os.makedirs("data/.cache", exist_ok=True)
    os.makedirs("data/.cache/figures", exist_ok=True)
    os.makedirs(COMMIT_FILE_SC, exist_ok=True)
    os.makedirs(COMMIT_DIFF_SC, exist_ok=True)
    os.makedirs(COMMIT_FUNCTION_SC, exist_ok=True)


def get_data(project: str) -> pl.DataFrame:
    return pl.read_csv(f"data/{project}.csv")


def preprocess(n: Optional[int] = None) -> List[Tuple[str, str]]:
    validate_structure()
    project = "ffmpeg"
    devign = get_data(project)
    devign = devign.sample(fraction=1.0, seed=42, shuffle=True)
    data_dir = Path(__file__).parent.parent / "data"
    project_dir = data_dir / f"repo/{project}"
    n = n if n is not None else len(devign)
    processed_commits = []

    for row in devign[:n].iter_rows(named=True):
        commit = row["sha_id"]
        project = row["project"]
        label = row["vulnerability"]

        for target_name, get_target in [
            ("commit_file_sc", get_commit_file_sc),
            ("commit_function_sc", get_commit_functions_sc),  # TODO finsih
            ("commit_diff_sc", get_commit_lines_sc),
        ]:
            skip_name = f"{target_name}/{project}_{commit}_skip.pkl"
            name = f"{target_name}/{project}_{commit}_{label}.pkl"
            if (data_dir / name).exists() or (data_dir / skip_name).exists():
                continue

            print(f"Processing {commit} {label}")
            checkout_commit(commit, project_dir)
            target_results = get_target(commit, project_dir)
            if len(target_results) == 0:
                os.system(f"touch {data_dir / skip_name}")
                continue

            # TODO Ensure that we leave with the same set of commits for each process.
            # currently not all the same commits are skipped.
            save_target(data_dir / name, target_results)
            processed_commits.append((target_results, commit))

    return processed_commits
