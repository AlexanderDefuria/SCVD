from pathlib import Path


def write_file(file_path: Path, content: str) -> None:
    """
    Write content to a file
    """
    with open(file_path, "w") as file:
        file.write(content)


def extract_cfg(project: Path) -> None:
    """
    Extract and save CFG using joern
    """
    pass
    # TODO implement this function
    # write_file(project / "edges..json", "CFG content")
