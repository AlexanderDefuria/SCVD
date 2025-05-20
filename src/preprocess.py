from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import pygit2 as git
import polars as pl
from pyjoern import parse_source
from src.utils import interact
import pickle
import os

VERBOSE = " > /dev/null 2>&1"
COMMIT_CPGS = Path("data/commit_cpgs")
COMMIT_DIFF_SC = Path("data/commit_diff_sc")
COMMIT_FUNCTION_SC = Path("data/commit_function_sc")
TEST = True


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
        print(f"Ignoring file: {filename}")
        return True
    return False


def get_diff(commit: str, project_dir: Path) -> git.Diff:
    repo = git.init_repository(project_dir)
    diff = repo.diff(commit + "^", commit, context_lines=0, interhunk_lines=0)
    assert isinstance(diff, git.Diff)
    return diff


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


def get_commit_file_cpgs(commit: str, project_dir: Path) -> List:
    """
    Get the functions that are added/modified/deleted in the commit
    If the commit is outside a function get the entire file.
    """
    # cfg is a graph of the given function.
    # cfg contains in/out edges between blocks.
    # Each block has a statement (source code line).
    # Ex:
    # >>> list(cpg['http_open_cnx_internal'].cfg.out_edges)[0][0].statements
    # [<Return: return location_changed;,returnlocation_changed;>]
    # Note: This currently filters to the CFG of the functions that are changed
    project_name = project_dir.name
    diff = get_diff(commit, project_dir)
    files, lines = get_commit_scope(diff, project_name)
    cpgs = []

    for file in files:
        # Get the CPG for the file
        # Get the functions in the file
        cfg = parse_source(project_dir / file)
        for function_name, function in cfg.items():
            if function.start_line is None or function.end_line is None:
                continue
            # Check if the function has changed lines, add if it does
            if not lines[file].isdisjoint(set(range(function.start_line, function.end_line + 1))):
                cpgs.append(function)

    return cpgs


def get_commit_functions_sc(commit: str, project_dir: Path) -> List[Dict]:
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
    return functions


def get_commit_lines_sc(commit: str, project_dir: Path) -> List[Dict]:
    project_name = project_dir.name
    diff = get_diff(commit, project_dir)
    commits = []

    # Get the files that are changed in the commit
    # Filter out non C/C++ files so we can parse everything.
    for obj in diff.deltas:
        obj: git.DiffDelta = obj
        file = Path(__file__).parent.parent / "data" / "repo" / project_name / obj.old_file.path
        if not ignore_file(file) and file.exists():
            with open(file, "r") as f:
                source_code = f.read()
                # Get the lines of code in the file
                lines_of_code = source_code.splitlines()
                for line in lines_of_code:
                    commits.append(
                        {
                            "commit": commit,
                            "file": file,
                            "line": line,
                        }
                    )

    return commits


def save_target(pickle_file: Path, target: List) -> None:
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
    os.makedirs(COMMIT_CPGS, exist_ok=True)
    os.makedirs(COMMIT_DIFF_SC, exist_ok=True)
    os.makedirs(COMMIT_FUNCTION_SC, exist_ok=True)


def get_data(project: str) -> pl.DataFrame:
    return pl.read_csv(f"data/{project}.csv")


def preprocess(scope: str, n: Optional[int] = None) -> List[Tuple[str, str]]:
    validate_structure()
    project = "ffmpeg"
    devign = get_data(project)
    devign = devign.sample(fraction=1.0, seed=42)
    data_dir = Path(__file__).parent.parent / "data"
    project_dir = data_dir / f"repo/{project}"
    n = n if n is not None else len(devign)
    processed_commits = []

    for row in devign[:n].iter_rows(named=True):
        commit = row["sha_id"]
        project = row["project"]
        label = row["vulnerability"]

        for target_name, get_target in [
            ("commit_cpgs", get_commit_file_cpgs),
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
            continue
            print(f"Error processing {commit} {label}: {e}")
            with open("data/errors.txt", "a") as f:  # Save the error to a file
                f.write(f"{commit} {label}: {e}\n")


    return processed_commits
