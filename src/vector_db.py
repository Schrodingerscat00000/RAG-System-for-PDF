import os
import shutil
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List
from src.config import CHROMA_DB_PATH
from src.embedding_model import get_embedding_model
from langchain_community.vectorstores.utils import filter_complex_metadata


def get_vector_store(embedding_model, collection_name: str = "rag_documents"):
    """
    Initializes and returns a ChromaDB vector store.
    If the database exists, it loads it; otherwise, it creates a new one.
    """
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Loading existing ChromaDB from {CHROMA_DB_PATH}...")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embedding_model,
            collection_name=collection_name
        )
    else:
        print(f"ChromaDB not found. It will be created when documents are added.")
        vector_store = Chroma(
            embedding_function=embedding_model,
            collection_name=collection_name,
            persist_directory=CHROMA_DB_PATH
        )
    return vector_store


def add_documents_to_vector_store(vector_store: Chroma, documents: List[Document]):
    """
    Adds documents to the ChromaDB vector store, filtering out complex metadata.
    """
    if not documents:
        print("No documents to add.")
        return

    print(f"Filtering and adding {len(documents)} child chunks to ChromaDB...")
    
    # This function is designed to remove metadata that Chroma can't handle
    # It's a safer way to prepare documents for ingestion
    cleaned_docs = filter_complex_metadata(documents)

    if not cleaned_docs:
        print("Error: No valid documents remained after filtering complex metadata.")
        return

    try:
        vector_store.add_documents(cleaned_docs)
        print(f"Successfully added {len(cleaned_docs)} documents to ChromaDB.")
    except Exception as e:
        print(f"Error adding documents to ChromaDB: {e}")


def get_retriever(vector_store: Chroma, k: int = 5):
    """Returns a retriever from the vector store."""
    # Increase k for more child chunks per query and set cosine similarity threshold if available
    search_kwargs = {"k": max(k, 10)}
    # If Chroma supports similarity threshold, add it here (pseudo-code):
    # search_kwargs["score_threshold"] = 0.7
    return vector_store.as_retriever(search_kwargs=search_kwargs)