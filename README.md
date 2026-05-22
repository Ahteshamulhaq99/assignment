# Visual Search at Scale

A production-grade, self-contained visual search engine for a fashion e-commerce platform. It leverages **Hugging Face `Marqo/fashion200k`** as its catalogue proxy, extracting semantic representations using **CLIP**, indexing them using **FAISS** (supporting Flat, HNSW, and IVF-PQ index designs), and explaining retrieval decisions using a **Local LLM (`Qwen2.5-1.5B-Instruct`)** to provide plain-text justifications based on aligned visual attributes.

## Features

- **Multimodal Embeddings**: Uses CLIP (`openai/clip-vit-large-patch14`) to extract L2-normalized image embeddings.
- **Scalable Indexing**: Built with FAISS, supporting scalable offline indexing via HNSW or IVF-PQ to handle millions of vectors efficiently.
- **LLM-Powered Grounded Explanations**: Uses an on-device LLM to generate plain-text, interpretable explanations of *why* an item was retrieved based on shared visual attributes (e.g., color, pattern, category, material).
- **FastAPI Backend**: Fully asynchronous, high-throughput REST API.

## 1. Project Directory Structure

```text
assigment/
├── frontend/
│   └── index.html           # Interactive visual search UI
├── src/
│   ├── __init__.py          # Marks src as a python package
│   ├── config.py            # Global paths, parameters, and attribute vocabularies
│   ├── data_loader.py       # Dataset streaming & local image cache storage
│   ├── embedder.py          # Singleton CLIP model wrapper (CUDA/CPU)
│   ├── indexer.py           # FAISS Index manager (Flat, HNSW, IVF)
│   ├── explainer.py         # Grounded explanation generator using Qwen
│   ├── server.py            # FastAPI application endpoints
│   ├── main.py              # Server uvicorn runner
│   └── rebuild_index.py     # CLI tool to rebuild the FAISS index
├── tests/
│   ├── __init__.py
│   ├── test_api.py          # Pytest API tests
│   └── run_tests.py         # Integration test suite
├── architecture.md          # 2-page system architecture write-up
├── requirements.txt         # Package dependencies list
├── postman_collection.json  # Postman collection for all endpoints
└── README.md                # This documentation
```

## 2. Environment Setup

### Prerequisites
*   **Python 3.11**
*   **CUDA-compatible GPU** (Recommended to run the CLIP model and Qwen LLM efficiently. Automatically falls back to CPU if unavailable, though explanations will use a basic fallback if LLM cannot be loaded).

### Setup Instructions
1.  **Clone / Open the Workspace Directory**:
    Navigate to the workspace root:
    ```bash
    cd assigment
    ```

2.  **Create and Activate a Virtual Environment**:
    ```bash
    python -m venv venv
    # On Windows (PowerShell):
    .\venv\Scripts\Activate.ps1
    # On Linux/macOS:
    source venv/bin/activate
    ```

3.  **Install Required Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 3. Running the Visual Search API and UI

Start the Uvicorn FastAPI server:
```bash
python -m src.main
```
The server will bind to `http://localhost:8000`. 

### Interactive UI
Navigate to **`http://localhost:8000/`** in your browser to test the visual search engine. You can upload an image, view the live index status, and inspect the ranked similarity results alongside the LLM's explanation for the match.

### Zero-Setup Instant Boot
When you start the server, it checks if a pre-built FAISS index exists on disk.
*   **If found**: It loads the index instantly and boots.
*   **If not found**: It automatically triggers a background bootstrap indexing task that streams 1,000 items from Hugging Face, saves their images locally, extracts embeddings, and creates the index. The server remains online, and the search API becomes fully functional in **10-15 seconds**.

## 4. Building & Scaling the Index

To build a larger index or customize the search configurations, you have two options:

### Option A: Command Line Interface (CLI)
You can use `src/rebuild_index.py` to index a custom subset or the full dataset:
```bash
# Index a subset of 5,000 items using HNSW graph index
python src/rebuild_index.py --limit 5000 --type hnsw

# Index 50,000 items using IVF-PQ (Quantized) index for low-RAM footprint
python src/rebuild_index.py --limit 50000 --type ivf
```

### Option B: REST API Endpoint
You can trigger a rebuild asynchronously in the background by calling `/index/rebuild` via a POST request. The frontend UI also reflects the live status of the background indexing task.

## 5. How Explanations Work (Model Grounding)

1.  **Visual Projection**: Rather than running a Generative Vision-Language Model directly on the raw images (which destroys request-response latency), the system projects the query image embedding and the retrieved product image embedding onto a vocabulary of fashion attributes in the shared CLIP space.
2.  **Intersection Analysis**: The system extracts the top matching attributes for both images to identify overlapping characteristics.
3.  **LLM Summarization**: An on-device local LLM (`Qwen2.5-1.5B-Instruct`) takes the shared attributes and generates a concise, natural language explanation highlighting the visual reasoning behind the match. This is completely grounded in CLIP's visual observations while delivering a friendly, human-readable response.
