from functools import cache
import os
import torch
from pathlib import Path
from typing import Any, List

from tqdm import tqdm
from transformers import Pipeline


def get_filenames_of_dir(directory: Path) -> List[str]:
    """
    Returns a list of absolute files paths of files in a given directory.

    Sorts by last modified time.
    """
    return sorted(
        [
            os.path.abspath(os.path.join(directory, f))
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ],
        key=os.path.getmtime,
    )


@cache
def get_available_device() -> str:
    """
    Get the available device for running the model
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"
    return "cpu"


def get_id(
    input_path: Path,
) -> str:
    """
    Generate a unique ID for a given type and hash
    """
    return f"{type}_{hash}"
