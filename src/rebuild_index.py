import os
import sys
import argparse
import logging
import time
from PIL import Image

# Add parent directory of rebuild_index.py to Python path to support direct execution
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import INDEX_TYPE, DATA_DIR
from src.embedder import CLIPEmbedder
from src.indexer import FashionIndexer
from src.data_loader import FashionDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Build or Rebuild the Visual Search Index from Hugging Face Fashion-200K.")
    parser.add_argument(
        "--limit", 
        type=int, 
        default=2000, 
        help="Number of items to fetch and index from HuggingFace (set to 0 or leave empty for full catalog)"
    )
    parser.add_argument(
        "--type", 
        type=str, 
        default=INDEX_TYPE, 
        choices=["flat", "hnsw", "ivf"],
        help="Type of FAISS index to build (flat, hnsw, ivf)"
    )
    args = parser.parse_args()
    
    limit = None if args.limit <= 0 else args.limit
    
    logger.info("=== Starting Visual Search Index Build ===")
    t_start = time.time()
    
    # 1. Initialize loader and stream data
    loader = FashionDataLoader()
    df_metadata = loader.fetch_and_store_subset(limit=limit)
    
    # 2. Load model
    logger.info("Initializing CLIP embedder...")
    embedder = CLIPEmbedder()
    
    # 3. Generate embeddings
    logger.info(f"Extracting image embeddings for {len(df_metadata)} items...")
    pil_images = []
    for idx, row in df_metadata.iterrows():
        img_path = DATA_DIR / row["image_path"]
        pil_images.append(Image.open(img_path))
        
    t_embed_start = time.time()
    embeddings = embedder.embed_images(pil_images, batch_size=64)
    t_embed = time.time() - t_embed_start
    logger.info(f"Embeddings extracted in {t_embed:.2f}s ({len(df_metadata) / t_embed:.2f} items/sec).")
    
    # 4. Build and save index
    indexer = FashionIndexer(dimension=embedder.get_embedding_dim(), index_type=args.type)
    indexer.build_index(embeddings, df_metadata)
    indexer.save_index()
    
    total_time = time.time() - t_start
    logger.info(f"=== Indexing Complete in {total_time:.2f}s! ===")
    logger.info(f"Total items indexed: {len(df_metadata)}")
    logger.info(f"Index type: {args.type}")

if __name__ == "__main__":
    main()
