from functools import cache
import hashlib
from pathlib import Path


@cache
def get_file_hash(filename: str | Path, algorithm="sha256", chunk_size=8192):
    """Calculate the hash of a file using the specified algorithm.

    Example usage:
    hash_value = get_file_hash('path/to/your/file.txt')

    Args:
        filename (str): Path to the file
        algorithm (str): Hash algorithm to use (default: sha256)
        chunk_size (int): Size of chunks to read (default: 8192 bytes)

    Returns:
        str: Hexadecimal representation of the hash
    """
    hash_obj = hashlib.new(algorithm)

    with open(filename, "rb") as f:
        while chunk := f.read(chunk_size):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()
