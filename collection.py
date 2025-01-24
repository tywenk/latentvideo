import enum
from textwrap import shorten
from typing import Tuple
from chromadb.api.types import OneOrMany
from chromadb.config import Settings
from chromadb.types import Metadata
from chromadb.utils.embedding_functions.open_clip_embedding_function import (
    OpenCLIPEmbeddingFunction,
)

import chromadb
from chromadb.utils.data_loaders import ImageLoader

from filehash import get_file_hash
from logger import Logger


class CollectionId(enum.Enum):
    AUDIO_EXTRACT = "ax"
    FRAME_EXTRACT = "fx"
    CAPTION_EXTRACT = "cx"


class Collection:
    """
    This class colacates multiple chroma db collections.

    One collection stores

    Uses Open CLIP for both text and image embeddings
    """

    def __init__(self, *, input_path: str, path="chromadb"):
        self.logger = Logger.setup(__name__)
        self._client = chromadb.PersistentClient(
            path=path, settings=Settings(anonymized_telemetry=False)
        )

        self.input_path = input_path

        self.img_embedding_function = OpenCLIPEmbeddingFunction()
        self.img_data_loader = ImageLoader()

        self.img_collection = self._client.get_or_create_collection(
            name="multimodal_db",
            embedding_function=self.img_embedding_function,
            data_loader=self.img_data_loader,
            # hsnw:space options:
            # L2 (Squared L2): Finding the closest vectors in terms of actual geometric distance in the embedding space
            # Inner Product: Making recommendations that consider both similarity and "importance" (as represented by vector magnitude)
            # Cosine Similarity: Comparing text embeddings where you want to match meaning regardless of text length
            metadata={"hnsw:space": "cosine", "hnsw:search_ef": 100},
        )
        self.logger.info(
            f"Got image collection: {self.img_collection.name} {self.img_collection.id}"
        )

        self.txt_collection = self._client.get_or_create_collection(
            name="textual_db",
            data_loader=self.img_data_loader,
            # hsnw:space options:
            # L2 (Squared L2): Finding the closest vectors in terms of actual geometric distance in the embedding space
            # Inner Product: Making recommendations that consider both similarity and "importance" (as represented by vector magnitude)
            # Cosine Similarity: Comparing text embeddings where you want to match meaning regardless of text length
            metadata={"hnsw:space": "cosine", "hnsw:search_ef": 100},
        )
        self.logger.info(
            f"Got text collection: {self.txt_collection.name} {self.txt_collection.id}"
        )

    def _get_col_id(self, num: int, kind: CollectionId) -> str:
        return shorten(f"{num}_{kind.value}_{get_file_hash(self.input_path)}", width=63)

    def add_image(
        self, kind: CollectionId, uris: list[str], metadatas: OneOrMany[Metadata]
    ):
        ids = [self._get_col_id(i, kind=kind) for i in range(len(uris))]
        self.img_collection.add(ids=ids, uris=uris, metadatas=metadatas)

    def add_text(
        self, kind: CollectionId, documents: list[str], metadatas: OneOrMany[Metadata]
    ):
        ids = [self._get_col_id(i, kind=kind) for i in range(len(documents))]
        self.txt_collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def search(
        self, query: str, n_results: int = 2
    ) -> Tuple[chromadb.QueryResult, chromadb.QueryResult]:
        self.logger.info(f"Search query in {self.img_collection.name}")
        img_res = self.img_collection.query(query_texts=[query], n_results=n_results)
        self.logger.info(f"Search query in {self.txt_collection.name}")
        txt_res = self.txt_collection.query(query_texts=[query], n_results=n_results)

        return img_res, txt_res
