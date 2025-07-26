import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import PDF_PATH, CHROMA_DB_PATH, K_CHILD_CHUNKS, K_PARENT_CHUNKS
from document_processor import extract_pdf_content, create_hybrid_chunks
from embedding_model import get_embedding_model
from vector_db import get_vector_store, add_documents_to_vector_store, get_retriever
from llm_model import get_llm
from rag_system import create_rag_chain, create_conversational_rag_chain

def setup_rag_pipeline(
    embedding_model_name: str = "bge-m3",
    llm_model_type: str = "cohere",
    llm_quantized: bool = True,
    use_conversation_memory: bool = False
):
    """
    Sets up the full RAG pipeline.
    """
    # 1. Text Extraction
    print("\n--- Step 1: Extracting PDF Content ---")
    documents = extract_pdf_content(PDF_PATH)
    if not documents:
        print("No documents extracted. Exiting.")
        return None, None, None

    # 2. Chunking Strategy (Parent-Child)
    print("\n--- Step 2: Creating Hybrid Chunks ---")
    chunk_data = create_hybrid_chunks(documents)
    parent_chunks = chunk_data["parent_chunks"]
    child_chunks_for_db = chunk_data["child_chunks_for_db"]
    parent_id_to_document = chunk_data["parent_id_to_document"]

    # 3. Embedding Model
    print("\n--- Step 3: Initializing Embedding Model ---")
    embedding_model = get_embedding_model(embedding_model_name)

    # 4. Vector Database (ChromaDB)
    print("\n--- Step 4: Setting up Vector Database ---")
    vector_store = get_vector_store(embedding_model)

    # Add documents if not already in DB (check by counting existing docs)
    # Note: Chroma's `get` can be slow for very large DBs. For small DBs, it's fine.
    # For robust check, you'd implement a versioning or checksum system.
    if vector_store._collection.count() == 0: # Checks if collection is empty
        add_documents_to_vector_store(vector_store, child_chunks_for_db)
    else:
        print(f"Vector store already contains {vector_store._collection.count()} documents. Skipping re-embedding.")

    retriever = get_retriever(vector_store, k=K_CHILD_CHUNKS)

    # 5. LLM
    print("\n--- Step 5: Initializing LLM ---")
    llm = get_llm(llm_model_type, llm_quantized)

    # 6. Create RAG Chain
    print("\n--- Step 6: Creating RAG Chain ---")
    if use_conversation_memory:
        rag_chain = create_conversational_rag_chain(llm, retriever, parent_id_to_document)
        print("Using Conversational RAG Chain with short-term memory.")
    else:
        rag_chain = create_rag_chain(llm, retriever, parent_id_to_document)
        print("Using Standard RAG Chain (no short-term memory).")


    return rag_chain, retriever, parent_id_to_document

def main():
    # --- Configuration Options ---
    # Choose your embedding model: "cohere", "bge-m3", "multilingual-e5"
    EMBED_MODEL = "bge-m3"

    # Choose your LLM: "cohere", "qwen2-7b-instruct"
    LLM_MODEL = "cohere"
    LLM_QUANTIZED = True # Only applicable for HuggingFace models like Qwen

    # Enable short-term memory for conversational experience
    USE_CONVERSATION_MEMORY = True
    # ---------------------------

    rag_chain, retriever, parent_id_to_document = setup_rag_pipeline(
        embedding_model_name=EMBED_MODEL,
        llm_model_type=LLM_MODEL,
        llm_quantized=LLM_QUANTIZED,
        use_conversation_memory=USE_CONVERSATION_MEMORY
    )

    if rag_chain is None:
        return

    print("\n--- RAG System Ready! Start asking questions. ---")
    print("Type 'exit' to quit.")

    chat_history = []

    while True:
        query = input("\nYour question (English/Bangla): ")
        if query.lower() == 'exit':
            break

        print("\nSearching and Generating Answer...")
        try:
            if USE_CONVERSATION_MEMORY:
                result = rag_chain.invoke({"input": query, "chat_history": chat_history})
                answer = result["answer"].get('answer', "Sorry, I couldn't find an answer.")
                source_documents = result["answer"].get("context", [])
                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": answer})
            else:
                answer = rag_chain.invoke(query)
                source_documents = retriever.invoke(query)

            print(f"\nAnswer: {answer}")

            print("\n--- Retrieved Sources ---")
            if source_documents:
                for i, doc in enumerate(source_documents):
                    source_info = f"Source {i+1}"
                    if 'parent_id' in doc.metadata:
                        source_info += f" (Parent ID: {doc.metadata['parent_id']})"
                    if 'page_number' in doc.metadata:
                        source_info += f" (Page: {doc.metadata['page_number']})"
                    print(source_info)
                    print(f"Content snippet: {doc.page_content[:200]}...")
            else:
                print("No relevant sources found.")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure your API keys are correct and models are loaded properly.")

if __name__ == "__main__":
    main()