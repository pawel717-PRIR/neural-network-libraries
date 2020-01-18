from pathlib import Path


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent


def get_datasets_path() -> Path:
    """Returns datasets folder."""
    return get_project_root().joinpath('datasets')
