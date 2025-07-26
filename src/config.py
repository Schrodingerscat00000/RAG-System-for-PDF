import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY") # Added Voyage API Key
HF_TOKEN = os.getenv("HF_TOKEN") # For HuggingFace models

# Tesseract OCR Path
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")

# Document Paths
PDF_PATH = "data/HSC26-Bangla1st-Paper.pdf"

# Chunking Parameters
PARENT_CHUNK_SIZE = 3000 # Example: Larger chunks for parent
CHILD_CHUNK_SIZE = 500   # Example: Smaller chunks for embeddings
CHUNK_OVERLAP = 100       # Overlap for child chunks

# Retrieval Parameters
K_CHILD_CHUNKS = 10     # Number of child chunks to retrieve
K_PARENT_CHUNKS = 3      # Number of parent chunks to return (if using parent-child)

# Vector DB
CHROMA_DB_PATH = "chroma_db"