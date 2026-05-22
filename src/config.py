import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "indexes"
INDEX_FILE = INDEX_DIR / "faiss_index.index"
METADATA_FILE = INDEX_DIR / "metadata.parquet"
ATTRIBUTES_FILE = INDEX_DIR / "attributes.json"

# Create directories if they do not exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Dataset & Model Settings
DATASET_NAME = "Marqo/fashion200k"
# We use standard CLIP model which is fast and handles multimodal retrieval well
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

# Embedding Dimension
EMBEDDING_DIM = 768  # CLIP-ViT-L/14 has 768 dimensions

# FAISS Index Configuration
# Choices: "flat" (exact L2/IP search), "hnsw" (high-speed graph-based), "ivf" (inverted file for compression/scale)
INDEX_TYPE = "hnsw"
M_PARAMETER = 32  # HNSW parameter (number of connections per node)
EF_CONSTRUCTION = 64  # HNSW parameter (depth of search during index creation)
EF_SEARCH = 32  # HNSW parameter (depth of search during query)

# Search Settings
DEFAULT_TOP_K = 10
SIMILARITY_THRESHOLD = 0.5  # Cosine similarity threshold for explanations

# Local LLM settings for premium explanations (optional)
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"



# Fashion Attribute Vocabulary (used by CLIP to ground explanations in model observations)
FASHION_ATTRIBUTES = {
    "color": [
        "black", "white", "grey", "red", "blue", "green", "yellow", "pink", 
        "purple", "brown", "beige", "orange", "gold", "silver", "multicolor", "navy"
    ],
    "pattern": [
        "solid", "striped", "floral", "plaid", "polka dot", "checkered", 
        "leopard", "graphic print", "lace", "patterned", "embroidered", "knit"
    ],
    "category": [
        "dress", "shirt", "t-shirt", "jacket", "coat", "pants", "jeans", 
        "skirt", "sweater", "hoodie", "shoes", "boots", "sandals", "bag", 
        "shorts", "suit", "blazer", "blouse", "romper", "cardigan"
    ],
    "neckline": [
        "v-neck", "round neck", "crew neck", "boat neck", "collar", 
        "off-the-shoulder", "sweetheart", "turtleneck", "hooded"
    ],
    "sleeve": [
        "sleeveless", "short sleeve", "long sleeve", "three-quarter sleeve", "strap"
    ],
    "material": [
        "denim", "leather", "cotton", "shiny satin", "matte leather", 
        "silk", "velvet", "wool", "linen", "suede", "mesh", "fur"
    ]
}
