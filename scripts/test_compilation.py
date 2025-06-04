from pathlib import Path
import polars as pl
import subprocess
import os

from src.utils import checkout_commit


if __name__ == "__main__":
    df = pl.read_csv("data/ffmpeg.csv")

    commits = df.sample(100)['sha_id'].to_list()
    project_dir = Path("data/repo/ffmpeg")
    compiled = 0
    for commit in commits:
        checkout_commit(commit,  project_dir, before=True)
        os.system(f"echo 'CXXFLAGS += /permissive' >> {project_dir}/Makefile")
        proc = subprocess.Popen(["bash", "configure"], stdout=subprocess.PIPE, cwd=project_dir)
        proc.communicate()  # Wait for the configure script to finish
        if proc.returncode != 0:
            print(f"Configuration failed for commit {commit}. Skipping compilation.")
            exit()
        proc = subprocess.Popen(["make"], stdout=subprocess.PIPE, cwd=project_dir,)
        proc.communicate()
        if proc.returncode != 0:
            print(f"Compilation failed for commit {commit}. Skipping to next commit.")
            continue
        compiled += 1

    print(f"Compiled: {compiled}")

