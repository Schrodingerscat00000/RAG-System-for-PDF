import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List, Dict, Any
from collections import defaultdict
# src/document_processor.py (Add this line at the top with other imports)
from src.config import PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE, CHUNK_OVERLAP

# Set Tesseract command environment variable for Unstructured
# This is crucial for Unstructured to find Tesseract
os.environ["TESSERACT_CMD"] = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
# Also set for pypdfium2 if used directly (Unstructured should handle it)
# import pypdfium2 as pdfium # if you decide to use it directly
# pdfium.set_pgm_dir(os.path.dirname(os.environ["TESSERACT_CMD"]))


def extract_pdf_content(pdf_path: str) -> List[Document]:
    """
    Extracts text and structure from a PDF using UnstructuredPDFLoader.
    Includes OCR for better Bengali text extraction.
    """
    print(f"Loading PDF from: {pdf_path}")
    # mode="elements" tries to preserve document structure
    # strategy="auto" attempts to use fast, hi_res, or OCR based on content
    # For Bengali, "ocr_only" might be safer initially if text quality is poor
    try:
        loader = UnstructuredPDFLoader(pdf_path, mode="elements", strategy="hi_res", languages=["ben", "eng"])
        # Ensure Tesseract is configured for Bengali and English
        # Unstructured typically picks up TESSERACT_CMD env var.
        # If it struggles with Bengali, you might need to manually specify languages:
        # loader = UnstructuredPDFLoader(pdf_path, mode="elements", strategy="hi_res", languages=["eng", "ben"])

        docs = loader.load()
        print(f"Extracted {len(docs)} elements (documents/chunks) from PDF.")
        return docs
    except Exception as e:
        print(f"Error extracting PDF with Unstructured: {e}")
        print("Falling back to simpler text extraction (may lose structure).")
        # Fallback to PyPDF if Unstructured fails completely, though this loses structure
        from pypdf import PdfReader
        text = ""
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or "" # extract_text might fail for scanned Bengali
        if not text:
             print("PyPDF also failed to extract text. Ensure Tesseract is installed and configured for OCR.")
             return []
        # Wrap fallback text in a Document
        return [Document(page_content=text, metadata={"source": pdf_path, "page": "all"})]


def create_hybrid_chunks(documents: List[Document]) -> Dict[str, Any]:
    """
    Aggregates extracted elements into a single document, then splits into parent and child chunks.
    """
    if not documents:
        return {
            "parent_chunks": [],
            "child_chunks_for_db": [],
            "parent_id_to_document": {},
        }

    # Aggregate all extracted elements into one large document
    full_text = "\n".join([doc.page_content for doc in documents])
    source = documents[0].metadata.get("source", "unknown")
    parent_doc = Document(page_content=full_text, metadata={"source": source, "page_number": "all"})

    # Parent chunking
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(PARENT_CHUNK_SIZE, 1000),
        chunk_overlap=0,
        separators=["\n\n\n", "\n\n", "\n", ". ", "! ","| ", "? ", " "],
        length_function=len,
    )
    parent_chunks = parent_splitter.split_documents([parent_doc])

    parent_id_to_document = {}
    child_chunks_for_db = []

    print(f"Created {len(parent_chunks)} parent chunks.")
    for idx, parent in enumerate(parent_chunks[:3]):
        print(f"Parent chunk {idx}: {parent.page_content[:200]}...")

    # Child chunking
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(CHILD_CHUNK_SIZE, 300),
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ","| ", "! ", "? ", " "],
        length_function=len,
    )

    child_id_counter = 0
    for i, parent_doc in enumerate(parent_chunks):
        parent_id = str(i)
        parent_doc.metadata["parent_id"] = parent_id
        parent_id_to_document[parent_id] = parent_doc
        child_docs = child_splitter.split_documents([parent_doc])
        for child_doc in child_docs:
            child_doc.metadata["parent_id"] = parent_id
            child_doc.metadata["child_id"] = str(child_id_counter)
            child_doc.metadata["source"] = parent_doc.metadata.get("source", "unknown")
            child_doc.metadata["page_number"] = parent_doc.metadata.get("page_number", None)
            child_chunks_for_db.append(child_doc)
            child_id_counter += 1

    print(f"Created {len(child_chunks_for_db)} child chunks for embedding.")
    for idx, child in enumerate(child_chunks_for_db[:3]):
        print(f"Child chunk {idx}: {child.page_content[:200]}...")

    return {
        "parent_chunks": parent_chunks,
        "child_chunks_for_db": child_chunks_for_db,
        "parent_id_to_document": parent_id_to_document,
    }

# Example of table extraction from unstructured output - requires specific parsing
def extract_and_format_tables(elements: List[Dict[str, Any]]) -> List[Document]:
    """
    Extracts and formats table content into digestible text.
    Unstructured.io returns table content usually as 'text_as_html' or 'text_as_markdown'.
    """
    table_docs = []
    for element in elements:
        if element.metadata.get("category") == "Table":
            # Prefer text_as_markdown if available and cleaner
            table_content = element.metadata.get("text_as_markdown") or element.page_content
            # You might need to further process this table_content to make it readable by LLM,
            # e.g., convert Markdown table to plain text rows.
            table_docs.append(Document(page_content=table_content, metadata=element.metadata))
    return table_docs