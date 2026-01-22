import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data_for_rag"  # Only use data_for_rag folder

# ChromaDB settings
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "booking_rag_collection"

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and efficient
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# API settings
API_HOST = "https://ai-booking-assistant-svqo.onrender.com"
API_PORT = 8000
TOP_K_RESULTS = 1

# Supported file types
SUPPORTED_PDF_EXTENSIONS = [".pdf"]
SUPPORTED_JSON_EXTENSIONS = [".json"]