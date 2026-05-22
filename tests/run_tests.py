import os
import sys
import unittest
from io import BytesIO
from fastapi.testclient import TestClient
from PIL import Image

# Ensure the root of the workspace is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.server import app
from src.config import EMBEDDING_DIM

class TestVisualSearchAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== Initializing TestClient and Bootstrapping Index ===")
        # The TestClient will trigger the startup_event when entering the context
        cls.client = TestClient(app)
        cls.client.__enter__()
        print("=== TestClient Initialized ===\n")

    @classmethod
    def tearDownClass(cls):
        print("\n=== Tearing Down TestClient ===")
        cls.client.__exit__(None, None, None)
        print("=== TestClient Teardown Complete ===\n")

    def test_1_read_root(self):
        print("Testing root endpoint...")
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("Visual Search", data["service"])
        print("Root endpoint test passed.")

    def test_2_index_status(self):
        print("Testing index status endpoint...")
        response = self.client.get("/index/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["index_loaded"])
        self.assertGreater(data["items_indexed"], 0)
        self.assertEqual(data["embedding_dimension"], EMBEDDING_DIM)
        print(f"Index status test passed. Items indexed: {data['items_indexed']}")

    def test_3_search_endpoint(self):
        print("Testing search endpoint with mock image...")
        # Create a simple red PIL image for testing
        img = Image.new("RGB", (224, 224), color="red")
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)
        
        # Send image to the /search endpoint
        response = self.client.post(
            "/search",
            files={"file": ("test_image.jpg", img_byte_arr, "image/jpeg")},
            data={"top_k": 3}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check structure
        self.assertIn("query_metadata", data)
        self.assertIn("results", data)
        self.assertGreater(len(data["results"]), 0)
        
        first_result = data["results"][0]
        self.assertIn("item_ID", first_result)
        self.assertIn("categories", first_result)
        self.assertIn("description", first_result)
        self.assertIn("image_url", first_result)
        self.assertIn("similarity_score", first_result)
        self.assertIn("explanation", first_result)
        
        print(f"Search endpoint test passed. Top result: {first_result['item_ID']} (Score: {first_result['similarity_score']})")
        print(f"Explanation: {first_result['explanation']}")

if __name__ == "__main__":
    unittest.main()
