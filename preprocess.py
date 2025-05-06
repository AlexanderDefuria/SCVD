import polars as pl
from pathlib import Path
import os



def checkout_commit(commit: str, project_dir: Path) -> None:
    """
    Checkout the commit using git
    git -C {project_dir} checkout {commit}
    """
    os.system(f"git -C {project_dir} checkout {commit}")


def extract_cpg(commit: str, project: Path) -> None:
    """
    Extract and save cpg using c2cpg.sh
    """
    os.system(f"c2cpg.sh -J-Xmx16G -o ./data/joern/{commit}.odb {project}")


def extract_analysis(commit: str, project: Path) -> None:
    "extract and save analysis graphs using joern"
    os.system


def validate_structure() -> None:
    assert Path("data").exists(), "data directory does not exist"
    assert Path("data/repo").exists(), "repo directory does not exist"
    assert Path("data/repo/ffmpeg").exists(), "ffmpeg directory does not exist"
    assert Path("data/repo/qemu").exists(), "qemu directory does not exist"
    assert Path("data/ffmpeg.csv").exists(), "ffmpeg.csv file does not exist"
    assert Path("data/qemu.csv").exists(), "qemu.csv file does not exist"
    os.makedirs("data/.cache", exist_ok=True)
    os.makedirs("data/joern", exist_ok=True)


def get_data(project: str) -> pl.DataFrame:
    return pl.read_csv(f"data/{project}.csv")


if __name__ == "__main__":
    validate_structure()

    project = "ffmpeg"
    devign = get_data(project)
    cpg_dir = Path(f"data/{project}/cpg")
    project_dir = Path(f"data/repo/{project}")

    for row in devign.iter_rows(named=True):
        commit = row["sha_id"]
        project = row["project"]
        label = row["vulnerability"]

        checkout_commit(commit, project_dir)
        extract_cpg(commit, project_dir)
        extract_analysis(commit, project_dir)
        exit()
        
        





    

    
    




