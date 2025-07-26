import os
#from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings # Import Embeddings base class
from typing import List
import voyageai # Import voyageai client
from langchain_cohere import CohereEmbeddings # Changed to import CohereEmbeddings
from langchain_core.embeddings import Embeddings
import torch

from src.config import COHERE_API_KEY, VOYAGE_API_KEY # Import VOYAGE_API_KEY

class VoyageEmbeddings(Embeddings):
    """Custom wrapper for Voyage AI embeddings to be compatible with LangChain."""
    def __init__(self, model_name: str = "voyage-3-large", input_type: str = "document"):
        if not VOYAGE_API_KEY:
            raise ValueError("VOYAGE_API_KEY is not set in environment variables.")
        self.client = voyageai.Client(api_key=VOYAGE_API_KEY)
        self.model_name = model_name
        self.input_type = input_type

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        # Voyage AI has token limits per request. For very long lists, you might need to batch.
        # Max tokens: 120K for voyage-multilingual-2 (as per API docs provided)
        # Max texts: 1,000
        # LangChain's add_documents handles batching, but be mindful of total token limits.
        try:
            result = self.client.embed(
                texts,
                model=self.model_name,
                input_type=self.input_type,
                truncation=True # Truncate if over length
            )
            return result.embeddings
        except Exception as e:
            print(f"Error embedding documents with Voyage AI: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a query."""
        try:
            result = self.client.embed(
                [text],
                model=self.model_name,
                input_type="query", # Always 'query' for query embedding
                truncation=True
            )
            return result.embeddings[0]
        except Exception as e:
            print(f"Error embedding query with Voyage AI: {e}")
            raise


def get_embedding_model(model_name: str = "bge-m3") -> Embeddings:
    """
    Returns the chosen embedding model.
    Options: 'cohere', 'voyage', 'bge-m3', 'multilingual-e5'.
    """
    if model_name == "cohere":
        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY is not set in environment variables.")
        print("Using Cohere Embeddings (embed-multilingual-v3.0).")
        return CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=COHERE_API_KEY)
    elif model_name == "voyage":
        print("Using Voyage AI Embeddings (voyage-3-large).")
        return VoyageEmbeddings(model_name="voyage-3-large") # Use our custom wrapper
    elif model_name == "bge-m3":
        print("Using BGE-M3 (BAAI/bge-m3) Embeddings.")
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    elif model_name == "multilingual-e5":
        print("Using Multilingual E5 (intfloat/multilingual-e5-large-instruct) Embeddings.")
        return HuggingFaceInstructEmbeddings(
            model_name="intfloat/multilingual-e5-large-instruct",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    else:
        raise ValueError(f"Unknown embedding model: {model_name}. Choose 'cohere', 'voyage', 'bge-m3', or 'multilingual-e5'.")