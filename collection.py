from chromadb.api.types import OneOrMany
from chromadb.types import Metadata
from chromadb.utils.embedding_functions.open_clip_embedding_function import (
    OpenCLIPEmbeddingFunction,
)

import chromadb
from chromadb.utils.data_loaders import ImageLoader

from logger import Logger


class Collection:
    """
    Collection of multimodal embeddings

    Uses Open CLIP for both text and image embeddings
    """

    def __init__(self, *, path="chromadb", name="multimodal_db"):
        self.logger = Logger.setup(__name__)
        self._client = chromadb.PersistentClient(path=path)

        self.embedding_function = OpenCLIPEmbeddingFunction()
        self.data_loader = ImageLoader()

        self.collection = self._client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_function,
            data_loader=self.data_loader,
            # hsnw:space options:
            # L2 (Squared L2): Finding the closest vectors in terms of actual geometric distance in the embedding space
            # Inner Product: Making recommendations that consider both similarity and "importance" (as represented by vector magnitude)
            # Cosine Similarity: Comparing text embeddings where you want to match meaning regardless of text length
            metadata={"hnsw:space": "cosine", "hnsw:search_ef": 100},
        )
        self.logger.info(f"Collection: {self.collection.name} {self.collection.id}")

    def add_image(self, ids: list[str], uris: list[str], metadatas: list[dict]):
        self.collection.add(ids=[], uris=[], metadatas=[])

    def add_text(
        self, ids: list[str], documents: list[str], metadatas: OneOrMany[Metadata]
    ):
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
