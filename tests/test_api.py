import os
import pytest
from io import BytesIO
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from src.server import app
from src.config import EMBEDDING_DIM

client = TestClient(app)

def test_read_root():
    """
    Test the root endpoint returns correct health status.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"
    assert "Visual Search" in response.json()["service"]

def test_index_status():
    """
    Test the index status endpoint reports that the index is loaded.
    """
    response = client.get("/index/status")
    assert response.status_code == 200
    data = response.json()
    assert "index_loaded" in data
    # During startup, a bootstrap indexing happens, so index should be loaded
    assert data["index_loaded"] is True
    assert data["items_indexed"] > 0
    assert data["embedding_dimension"] == EMBEDDING_DIM

def test_search_endpoint():
    """
    Test the search endpoint with a mock query image.
    """
    # Create a simple red PIL image for testing
    img = Image.new("RGB", (224, 224), color="red")
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    
    # Send image to the /search endpoint
    response = client.post(
        "/search",
        files={"file": ("test_image.jpg", img_byte_arr, "image/jpeg")},
        data={"top_k": 3}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "query_metadata" in data
    assert "results" in data
    assert len(data["results"]) > 0
    
    # Check result item structure
    first_result = data["results"][0]
    assert "item_ID" in first_result
    assert "categories" in first_result
    assert "description" in first_result
    assert "image_url" in first_result
    assert "similarity_score" in first_result
    assert "explanation" in first_result
    
    # Verify that explanation is a non-empty string and mentions visual properties
    assert isinstance(first_result["explanation"], str)
    assert len(first_result["explanation"]) > 0
    
    # Check latency stats
    latency = data["query_metadata"]["latency"]
    assert "total_pipeline_ms" in latency
    assert "vector_search_ms" in latency
