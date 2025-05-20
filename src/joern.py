
import os
import subprocess

from polars.catalog.unity import client



def create_joern_project(project_name: str, project_path: str) -> None:
    """
    Create a Joern project with the given name and path.
    Args:
        project_name (str): The name of the Joern project.
        project_path (str): The path where the Joern project will be created.
    """
    try:
        # Check if the Joern CLI is installed
        subprocess.run(["joern", "--version"], check=True)
    except FileNotFoundError:
        raise RuntimeError("Joern CLI is not installed. Please install it first.")

    # Create the Joern project
    os.makedirs(project_path, exist_ok=True)


    # TODO


