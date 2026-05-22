import os
import faiss
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple
from src.config import (
    INDEX_FILE, METADATA_FILE, EMBEDDING_DIM, INDEX_TYPE, 
    M_PARAMETER, EF_CONSTRUCTION, EF_SEARCH
)

logger = logging.getLogger(__name__)

class FashionIndexer:
    def __init__(self, dimension: int = EMBEDDING_DIM, index_type: str = INDEX_TYPE):
        self.dimension = dimension
        self.index_type = index_type.lower()
        self.index = None
        self.metadata = pd.DataFrame(columns=["item_ID", "category1", "category2", "category3", "text", "image_path"])
        
    def _create_empty_index(self, num_elements: int = 0) -> faiss.Index:
        """
        Creates an empty FAISS index based on configuration.
        """
        if self.index_type == "flat":
            logger.info("Creating a Flat Inner Product index.")
            return faiss.IndexFlatIP(self.dimension)
            
        elif self.index_type == "hnsw":
            logger.info(f"Creating an HNSW Flat Inner Product index (M={M_PARAMETER}, efConstruction={EF_CONSTRUCTION}).")
            # IndexHNSWFlat uses Inner Product metric
            index = faiss.IndexHNSWFlat(self.dimension, M_PARAMETER, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = EF_CONSTRUCTION
            index.hnsw.efSearch = EF_SEARCH
            return index
            
        elif self.index_type == "ivf":
            # For 2 million scale, we use IVF-PQ (Inverted File with Product Quantization)
            # We configure a coarse quantizer (IndexFlatIP) and train it on a subset
            # IVF search requires training, which will be executed in build_index
            logger.info("Initializing IVF-PQ index (requires training).")
            # We cluster into sqrt(N) centroids. For 2M, ~1024 or 2048 is standard.
            # For smaller scale test, we use 64 centroids.
            nlist = 1024 if num_elements >= 50000 else 64
            quantizer = faiss.IndexFlatIP(self.dimension)
            # PQ8: Quantize 512-dim vector into 64 bytes (8 bits per sub-vector)
            m = 64  # number of subquantizers
            # 8 bits per subquantizer (standard)
            index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
            return index
            
        else:
            logger.warning(f"Unknown index type: {self.index_type}. Defaulting to Flat.")
            return faiss.IndexFlatIP(self.dimension)

    def is_indexed(self) -> bool:
        """
        Checks if the index and metadata files exist on disk.
        """
        return os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE)

    def load_index(self):
        """
        Loads the FAISS index and metadata registry from disk.
        """
        if not self.is_indexed():
            raise FileNotFoundError("Index or Metadata file not found on disk.")
            
        logger.info(f"Loading FAISS index from {INDEX_FILE}...")
        self.index = faiss.read_index(str(INDEX_FILE))
        
        # Set HNSW search depth parameters if it's HNSW
        if isinstance(self.index, faiss.IndexHNSWFlat):
            self.index.hnsw.efSearch = EF_SEARCH
            
        logger.info(f"Loading metadata registry from {METADATA_FILE}...")
        self.metadata = pd.read_parquet(METADATA_FILE)
        
        logger.info(f"Index loaded. Contains {self.index.ntotal} vectors matching {len(self.metadata)} metadata records.")

    def save_index(self):
        """
        Saves the FAISS index and metadata registry to disk.
        """
        if self.index is None:
            raise ValueError("No index to save.")
            
        # Ensure index directory exists
        os.makedirs(INDEX_FILE.parent, exist_ok=True)
        
        logger.info(f"Saving FAISS index to {INDEX_FILE}...")
        faiss.write_index(self.index, str(INDEX_FILE))
        
        logger.info(f"Saving metadata registry to {METADATA_FILE}...")
        self.metadata.to_parquet(METADATA_FILE, index=False)
        
        logger.info("Index and metadata saved successfully.")

    def build_index(self, embeddings: np.ndarray, df_metadata: pd.DataFrame):
        """
        Builds the FAISS index from scratch with the given embeddings and metadata.
        """
        num_elements = len(embeddings)
        logger.info(f"Building index with {num_elements} vectors of dimension {self.dimension}...")
        
        # L2-normalize embeddings to ensure Inner Product = Cosine Similarity
        # (Assuming embeddings are already normalized by the embedder, but we check/force it here)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
        
        # Create empty index
        index = self._create_empty_index(num_elements)
        
        # IVF Index training is required
        if isinstance(index, faiss.IndexIVFPQ):
            if num_elements < index.nlist * 39:
                logger.warning(
                    f"Too few elements ({num_elements}) to train IVF index with {index.nlist} centroids. "
                    "Falling back to HNSW Flat for better retrieval quality on small data."
                )
                self.index_type = "hnsw"
                index = self._create_empty_index(num_elements)
            else:
                logger.info(f"Training IVF-PQ index on {num_elements} vectors...")
                index.train(embeddings.astype("float32"))
                logger.info("IVF-PQ training complete.")
                
        # Add vectors
        logger.info("Adding vectors to the FAISS index...")
        index.add(embeddings.astype("float32"))
        
        self.index = index
        self.metadata = df_metadata.copy()
        
        # Record the internal index IDs in metadata for mapping search results
        self.metadata["index_id"] = np.arange(len(df_metadata))
        
        logger.info(f"Index built successfully. Total vectors indexed: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """
        Searches the FAISS index for the k most similar items.
        Returns a list of dictionaries with matching metadata and distance scores.
        """
        if self.index is None:
            raise ValueError("Index is not loaded or built.")
            
        # Ensure query is L2 normalized
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
            
        # Reshape to (1, dim) if necessary
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Search the index
        # For inner product on normalized vectors, similarity = cosine similarity (range -1 to 1)
        distances, indices = self.index.search(query_embedding.astype("float32"), k)
        
        results = []
        for score, idx in zip(distances[0], indices[0]):
            # FAISS returns -1 for empty search slots (if dataset size < k)
            if idx == -1 or idx >= len(self.metadata):
                continue
                
            # Retrieve metadata row
            meta_row = self.metadata.iloc[idx]
            
            results.append({
                "item_ID": meta_row["item_ID"],
                "category1": meta_row["category1"],
                "category2": meta_row["category2"],
                "category3": meta_row["category3"],
                "text": meta_row["text"],
                "image_path": meta_row["image_path"],
                "score": float(score)  # Cosine similarity
            })
            
        return results
