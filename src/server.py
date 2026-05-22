import os
import time
import logging
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import pandas as pd

from pathlib import Path
from src.config import DATA_DIR, DEFAULT_TOP_K, INDEX_TYPE
from src.embedder import CLIPEmbedder
from src.indexer import FashionIndexer
from src.explainer import ExplanationEngine
from src.data_loader import FashionDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Visual Search at Scale API",
    description="Production-ready visual search engine for fashion e-commerce operating at scale.",
    version="1.0.0"
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (catalog images)
# So image path 'data/catalog_images/123.jpg' is served at '/static/catalog_images/123.jpg'
app.mount("/static", StaticFiles(directory=str(DATA_DIR)), name="static")

# Resolve frontend directory
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Core components (Singletons, lazy-loaded on startup)
embedder = None
indexer = None
explainer = None

# Global state to track indexing tasks
indexing_status = {
    "is_indexing": False,
    "progress": "idle",
    "items_indexed": 0,
    "last_error": None
}

def perform_indexing_background(limit: int, index_type: str):
    """
    Background task to stream data, embed images, and build the FAISS index.
    """
    global embedder, indexer, explainer, indexing_status
    indexing_status["is_indexing"] = True
    indexing_status["progress"] = "fetching_data"
    indexing_status["last_error"] = None
    
    try:
        logger.info(f"Starting background indexing of {limit} items using {index_type} index...")
        loader = FashionDataLoader()
        
        # 1. Fetch data and save images to disk
        df_metadata = loader.fetch_and_store_subset(limit=limit)
        
        # 2. Extract embeddings
        indexing_status["progress"] = "extracting_embeddings"
        logger.info("Extracting image embeddings using CLIP...")
        
        pil_images = []
        for idx, row in df_metadata.iterrows():
            img_path = DATA_DIR / row["image_path"]
            pil_images.append(Image.open(img_path))
            
        embeddings = embedder.embed_images(pil_images, batch_size=64)
        
        # 3. Build index
        indexing_status["progress"] = "building_index"
        logger.info("Building FAISS index...")
        
        new_indexer = FashionIndexer(dimension=embedder.get_embedding_dim(), index_type=index_type)
        new_indexer.build_index(embeddings, df_metadata)
        new_indexer.save_index()
        
        # 4. Swap active indexer
        indexer = new_indexer
        indexing_status["items_indexed"] = len(df_metadata)
        indexing_status["progress"] = "complete"
        logger.info(f"Background indexing completed. Indexed {len(df_metadata)} items.")
        
    except Exception as e:
        logger.error(f"Error during background indexing: {e}", exc_info=True)
        indexing_status["progress"] = "failed"
        indexing_status["last_error"] = str(e)
    finally:
        indexing_status["is_indexing"] = False

@app.on_event("startup")
async def startup_event():
    """
    Initializes models and loads or bootstraps the search index on startup.
    """
    global embedder, indexer, explainer
    
    logger.info("Initializing visual search system...")
    
    # 1. Load CLIP Model
    embedder = CLIPEmbedder()
    
    # 2. Initialize Explainer
    explainer = ExplanationEngine(embedder)
    
    # 3. Load or Bootstrap Indexer
    indexer = FashionIndexer(dimension=embedder.get_embedding_dim(), index_type=INDEX_TYPE)
    
    if indexer.is_indexed():
        logger.info("Pre-built index files found. Loading index...")
        indexer.load_index()
    else:
        logger.warning("No pre-built index found. Running bootstrap indexing (1,000 items) to guarantee API functionality...")
        # Run synchronously on startup to ensure API works immediately
        perform_indexing_background(limit=1000, index_type=INDEX_TYPE)


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """
    Serves the frontend testing UI.
    """
    html_path = FRONTEND_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend UI not found.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.get("/index/status")
def get_index_status():
    """
    Returns the status and metadata of the visual search index.
    """
    global indexer, indexing_status
    
    if indexer is None or indexer.index is None:
        return {
            "index_loaded": False,
            "indexing_status": indexing_status,
            "message": "Index is not loaded. Try triggering a rebuild."
        }
        
    num_vectors = indexer.index.ntotal
    index_type_class = indexer.index.__class__.__name__
    
    # Estimate memory footprint
    # Raw vector footprint
    raw_bytes = num_vectors * indexer.dimension * 4
    raw_mb = raw_bytes / (1024 * 1024)
    
    return {
        "index_loaded": True,
        "index_type_class": index_type_class,
        "configured_type": indexer.index_type,
        "items_indexed": num_vectors,
        "embedding_dimension": indexer.dimension,
        "approximate_memory_mb": round(raw_mb * 1.5, 2),  # Including graph overhead factor
        "indexing_status": indexing_status
    }

@app.post("/index/rebuild")
def rebuild_index(
    background_tasks: BackgroundTasks,
    limit: int = Form(2000, description="Number of items to index from HuggingFace"),
    index_type: str = Form(INDEX_TYPE, description="Type of FAISS index to build: flat, hnsw, ivf")
):
    """
    Triggers an asynchronous rebuild of the visual search index.
    """
    global indexing_status
    
    if indexing_status["is_indexing"]:
        raise HTTPException(status_code=400, detail="Indexing task is already in progress.")
        
    background_tasks.add_task(perform_indexing_background, limit, index_type)
    return {
        "message": "Indexing started in background.",
        "limit": limit,
        "index_type": index_type
    }

@app.post("/search")
async def search_similar_products(
    file: UploadFile = File(..., description="Query image to find matches for"),
    top_k: int = Form(DEFAULT_TOP_K, description="Number of matches to return")
):
    """
    Exposes visual search endpoint.
    Accepts an uploaded image, extracts its CLIP embedding, queries FAISS,
    and returns ranked visual similarities with explainable justifications.
    """
    global embedder, indexer, explainer
    
    if indexer is None or indexer.index is None:
        raise HTTPException(status_code=503, detail="Search index is not initialized yet. Please wait or trigger a rebuild.")
        
    t_start = time.time()
    
    # 1. Load uploaded image
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to process uploaded file: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file format.")
        
    # 2. Extract CLIP embedding
    try:
        t_embed_start = time.time()
        query_embedding = embedder.embed_images(image)
        t_embed = time.time() - t_embed_start
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process image features.")
        
    # 3. Query FAISS index
    try:
        t_search_start = time.time()
        search_results = indexer.search(query_embedding, k=top_k)
        t_search = time.time() - t_search_start
    except Exception as e:
        logger.error(f"FAISS search failed: {e}")
        raise HTTPException(status_code=500, detail="Vector search failed.")
        
    # 4. Generate visual explanations and construct absolute URLs
    t_explain_start = time.time()
    annotated_results = []
    
    # Load database embeddings for explanation references
    # (In FAISS, we can reconstruct vectors or compute on-the-fly,
    # but here we can just query the index to get product embeddings
    # or reconstruct them. To get the product embedding, we can reconstruct it from the index)
    for res in search_results:
        # Reconstruct the vector from the index if possible
        # Some indexes do not support direct reconstruction (e.g. IVF-PQ),
        # in which case we can fall back to general similarity explanation
        # or embed the retrieved images on the fly for explanation,
        # or load from a pre-saved array.
        # Since we serve statically, we can just load the saved image and embed it,
        # or reconstruct it. Reconstructing from HNSW or Flat is easy: index.reconstruct(id).
        # Let's do a try-except to reconstruct the vector, and if it fails,
        # we can embed the product image on the fly! This is extremely robust.
        prod_emb = None
        idx_id = indexer.metadata[indexer.metadata["item_ID"] == res["item_ID"]]["index_id"].values[0]
        try:
            prod_emb = indexer.index.reconstruct(int(idx_id))
        except Exception:
            # Fallback: embed the static image from disk on the fly
            try:
                prod_img_path = DATA_DIR / res["image_path"]
                prod_img = Image.open(prod_img_path).convert("RGB")
                prod_emb = embedder.embed_images(prod_img).flatten()
            except Exception as img_err:
                logger.error(f"Failed to load product image for embedding: {img_err}")
                
        if prod_emb is None:
            # Fallback to query embedding so that explanation engine doesn't crash
            prod_emb = query_embedding.flatten()
            
        explanation = explainer.generate_explanation(
            query_emb=query_embedding.flatten(),
            product_emb=prod_emb,
            product_metadata=res,
            score=res["score"]
        )
        
        # Build relative image URL
        # e.g., /static/catalog_images/123.jpg
        # The client will prepend the host
        image_url = f"/static/{res['image_path']}"
        
        annotated_results.append({
            "item_ID": res["item_ID"],
            "categories": {
                "level1": res["category1"],
                "level2": res["category2"],
                "level3": res["category3"]
            },
            "description": res["text"],
            "image_url": image_url,
            "similarity_score": round(res["score"], 4),
            "explanation": explanation
        })
        
    t_explain = time.time() - t_explain_start
    total_time = time.time() - t_start
    
    return {
        "query_metadata": {
            "top_k": top_k,
            "latency": {
                "embedding_generation_ms": round(t_embed * 1000, 2),
                "vector_search_ms": round(t_search * 1000, 2),
                "explanation_generation_ms": round(t_explain * 1000, 2),
                "total_pipeline_ms": round(total_time * 1000, 2)
            }
        },
        "results": annotated_results
    }
