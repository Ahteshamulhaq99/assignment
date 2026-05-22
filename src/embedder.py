import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import List, Union, Dict
import logging
from src.config import CLIP_MODEL_NAME

logger = logging.getLogger(__name__)

class CLIPEmbedder:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CLIPEmbedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, device: str = None):
        if self._initialized:
            return
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing CLIPEmbedder on device: {self.device}")
        
        # Load CLIP model and processor
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        self.embedding_dim = self.model.config.projection_dim
        self._initialized = True
        logger.info("CLIPEmbedder initialized successfully.")

    @torch.no_grad()
    def embed_images(self, images: Union[Image.Image, List[Image.Image]], batch_size: int = 32) -> np.ndarray:
        """
        Extract L2-normalized image embeddings for one or more PIL images.
        """
        if isinstance(images, Image.Image):
            images = [images]
            
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            # Extract vision features
            image_features = self.model.get_image_features(**inputs)
            if hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            # Normalize embeddings to unit sphere
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(image_features.cpu().numpy())
            
        return np.vstack(all_embeddings)

    @torch.no_grad()
    def embed_texts(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Extract L2-normalized text embeddings for one or more text queries.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            # Extract text features
            text_features = self.model.get_text_features(**inputs)
            if hasattr(text_features, "pooler_output"):
                text_features = text_features.pooler_output
            # Normalize embeddings to unit sphere
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(text_features.cpu().numpy())
            
        return np.vstack(all_embeddings)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim
