from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from typing import Any, Dict, List


class ParentChildRetriever(BaseRetriever):
    """
    A custom retriever that retrieves parent documents based on child chunk retrieval.
    """
    # Allow arbitrary types for base_retriever
    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_retriever: Any
    parent_id_to_document: Dict[str, Document]
    top_k: int = 5  # Increase to retrieve more parent docs for better context

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve child chunks and map them to parent documents."""
        child_chunks = self.base_retriever.invoke(query)
        parent_ids = set()
        for doc in child_chunks:
            if doc.metadata and 'parent_id' in doc.metadata:
                parent_ids.add(doc.metadata['parent_id'])
        parent_docs = [
            self.parent_id_to_document[pid]
            for pid in parent_ids
            if pid in self.parent_id_to_document
        ]
        print(f"Retrieved {len(child_chunks)} child chunks. Found {len(parent_docs)} parent docs.")
        # Sort parent docs by their original order (parent_id as int)
        parent_docs_sorted = sorted(parent_docs, key=lambda d: int(d.metadata['parent_id']))
        return parent_docs_sorted[:self.top_k]


def create_rag_chain(llm, retriever, parent_id_to_document: Dict[str, Document]):
    """
    Creates the RAG chain using LangChain.
    Implements the parent-child retrieval strategy.
    """
    # 1. Prompt for the LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant for a textbook. "
         "Answer based *only* on the provided context. "
         "If not found, state that fact. Context: {context}"),
        ("human", "{input}"),
    ])

    # 2. Document combining chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 3. Functional retriever wrapper for standard RAG
    def _parent_child_retriever_func(query: str) -> List[Document]:
        retriever_fn = ParentChildRetriever(base_retriever=retriever, parent_id_to_document=parent_id_to_document)
        return retriever_fn.invoke(query)

    # 4. Build the chain
    rag_chain = (
        {"context": _parent_child_retriever_func, "input": RunnablePassthrough()}
        | document_chain
        | StrOutputParser()
    )
    return rag_chain


# For Short-Term Memory (Conversation History)
from langchain.memory import ConversationBufferWindowMemory

def create_conversational_rag_chain(llm, retriever, parent_id_to_document: Dict[str, Document]):
    """
    Creates a conversational RAG chain with short-term memory (chat history).
    """
    # Wrap base retriever in custom class
    custom_retriever = ParentChildRetriever(base_retriever=retriever, parent_id_to_document=parent_id_to_document)

    # 1. Contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, custom_retriever, contextualize_q_prompt
    )

    # 2. Answer question prompt
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, just say "
        "that you don't know. Use three sentences maximum and keep the "
        "answer concise. Context: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 3. Create stuff documents chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 4. Create retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 5. Add a passthrough for the sources
    def get_sources(result):
        if 'source_documents' in result:
            return result['source_documents']
        return []

    rag_chain_with_sources = RunnablePassthrough.assign(
        answer=rag_chain,
        source_documents=RunnableLambda(get_sources)
    )

    return rag_chain_with_sources