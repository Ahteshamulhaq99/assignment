import os
import sys
import uvicorn
import logging

# Add parent directory of main.py to Python path to support direct execution
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Visual Search Engine Server...")
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=False)
