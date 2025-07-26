from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from src.main import setup_rag_pipeline # Import the setup function
from langchain_core.messages import HumanMessage, AIMessage

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual RAG API",
    description="A simple REST API for the Multilingual RAG system.",
    version="1.0.0",
)

# Global variables to hold the RAG chain and history
rag_chain = None
# This simple setup might not scale well for multiple concurrent users
# For production, you'd manage session-specific history more robustly (e.g., Redis)
chat_history = [] # For a single user session example

class QueryRequest(BaseModel):
    query: str
    # If you want to enable/disable conversation memory per request
    # use_history: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: list = []
    chat_history: list = [] # To show the updated history

@app.on_event("startup")
async def startup_event():
    """
    Load the RAG pipeline when the FastAPI application starts.
    """
    print("Loading RAG pipeline on startup...")
    global rag_chain # Declare global to modify
    # For simplicity, we hardcode the config for the API startup.
    # You might want to pass these via environment variables or another config file.
    rag_chain, retriever_ref, parent_id_to_document_ref = setup_rag_pipeline(
        embedding_model_name="voyage", # Or "bge-m3", "multilingual-e5"
        llm_model_type="cohere",       # Or "qwen2-7b-instruct"
        llm_quantized=True,            # Set to False if running large models without quantization
        use_conversation_memory=True   # API will maintain conversation memory
    )
    if rag_chain is None:
        raise RuntimeError("Failed to set up RAG pipeline. Check logs for errors.")
    print("RAG pipeline loaded successfully!")

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Endpoint to send a query to the RAG system and get an answer.
    """
    global chat_history
    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized.")

    try:
        # LangChain's ConversationalRetrievalChain handles chat_history internally if memory is set up
        # We just need to pass the current question
        result = rag_chain.invoke({"question": request.query, "chat_history": chat_history})

        answer = result["answer"]
        source_documents = result.get("source_documents", [])

        # Update chat history for the next turn
        chat_history.append(HumanMessage(content=request.query))
        chat_history.append(AIMessage(content=answer))

        # Prepare source information for the response
        sources_info = []
        for doc in source_documents:
            sources_info.append({
                "parent_id": doc.metadata.get("parent_id", "N/A"),
                "page_number": f"{doc.metadata.get('page_number_start', 'N/A')}-{doc.metadata.get('page_number_end', 'N/A')}",
                "content_snippet": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            })

        return QueryResponse(answer=answer, sources=sources_info, chat_history=[m.dict() for m in chat_history])

    except Exception as e:
        print(f"Error during RAG processing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# To run the API:
# uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# (Use --reload during development for auto-reloading)