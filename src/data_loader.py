import os
import datasets
from PIL import Image
import pandas as pd
import logging
from typing import Generator, Dict, Any, List
from src.config import DATASET_NAME, DATA_DIR

logger = logging.getLogger(__name__)

# Directory to store catalog images
IMAGE_DIR = DATA_DIR / "catalog_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

class FashionDataLoader:
    def __init__(self, dataset_name: str = DATASET_NAME):
        self.dataset_name = dataset_name
        logger.info(f"Initialized FashionDataLoader for dataset: {dataset_name}")

    def stream_dataset(self, limit: int = None) -> Generator[Dict[str, Any], None, None]:
        """
        Stream dataset samples from Hugging Face without downloading the entire 4.2GB file.
        Yields dictionaries containing:
            - 'item_ID': unique identifier
            - 'image': PIL Image
            - 'category1': general category
            - 'category2': subcategory
            - 'category3': detailed description/category
            - 'text': detailed text description
        """
        logger.info(f"Streaming dataset (limit={limit if limit else 'None'})...")
        try:
            # We load the dataset builder to get split info, and load split='data'
            # streaming=True allows lazy loading of Parquet files
            ds = datasets.load_dataset(self.dataset_name, split="data", streaming=True)
            
            count = 0
            for item in ds:
                if limit is not None and count >= limit:
                    break
                
                # Yield dataset item
                yield {
                    "item_ID": item["item_ID"],
                    "image": item["image"],
                    "category1": item.get("category1", ""),
                    "category2": item.get("category2", ""),
                    "category3": item.get("category3", ""),
                    "text": item.get("text", "")
                }
                count += 1
                
        except Exception as e:
            logger.error(f"Error streaming dataset: {e}")
            raise e

    def save_image_to_disk(self, image: Image.Image, item_id: str) -> str:
        """
        Saves a catalog image to disk as a JPEG, returning its relative path.
        """
        # Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        file_path = IMAGE_DIR / f"{item_id}.jpg"
        if not file_path.exists():
            image.save(file_path, "JPEG", quality=85)
        return str(file_path.relative_to(DATA_DIR))

    def fetch_and_store_subset(self, limit: int = 1000) -> pd.DataFrame:
        """
        Fetches a subset of the dataset, saves the images to disk, and returns
        a metadata DataFrame. Useful for initialization or offline indexing.
        """
        metadata_list = []
        logger.info(f"Fetching and storing subset of size: {limit}")
        
        for item in self.stream_dataset(limit=limit):
            item_id = item["item_ID"]
            image = item["image"]
            
            # Save image to disk
            rel_image_path = self.save_image_to_disk(image, item_id)
            
            metadata_list.append({
                "item_ID": item_id,
                "category1": item["category1"],
                "category2": item["category2"],
                "category3": item["category3"],
                "text": item["text"],
                "image_path": rel_image_path
            })
            
            if len(metadata_list) % 500 == 0:
                logger.info(f"Processed {len(metadata_list)} items...")
                
        df = pd.DataFrame(metadata_list)
        logger.info(f"Successfully processed {len(df)} items.")
        return df
