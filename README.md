
# Multilingual RAG System for PDF Document Corpus

This project implements a basic Retrieval-Augmented Generation (RAG) system capable of understanding and responding to queries in both English and Bengali. The system retrieves relevant information from a PDF document corpus and generates answers grounded in the retrieved content, maintaining a conversational memory.

---

## 🚀 Objective

The primary objective is to build a foundational RAG application that:
* Accepts user queries in English and Bengali.
* Retrieves relevant document chunks from a small knowledge base (a PDF document).
* Generates answers based on the retrieved information.
* Maintains short-term conversational memory.

---

## 🛠️ Used Tools, Libraries, and Packages

The following key libraries and tools are utilized in this project, managed via `requirements.txt`:

* **`Python 3.11/3.12`**: The primary programming language.
* **`langchain` (`==0.1.17`)**: Core framework for building LLM applications. Provides abstractions for chains, retrievers, and memory.
* **`langchain-core` (`==0.1.48`)**: Fundamental abstractions and runtime for LangChain.
* **`langchain-community` (`==0.0.36`)**: Integrations for various LLMs, retrievers, and other components.
* **`langchain-chroma` (`==0.2.5`)**: LangChain integration for ChromaDB.
* **`langchain-cohere` (`==0.1.0`)**: LangChain integration for Cohere models (both LLM and Embeddings).
* **`pydantic` (`==1.10.13`)**: Data validation library used by many LangChain components. Pinned to v1 for compatibility.
* **`cohere` (`==4.5.1`)**: Python SDK for interacting with Cohere's API (LLM and Embeddings).
* **`voyageai`**: (Initially used for embeddings, but replaced by Cohere due to billing limitations.)
* **`pypdfium2`**: A binding for PDFium, used by `unstructured` for efficient PDF processing.
* **`unstructured[all-docs]`**: A powerful library for parsing and extracting content from various document types, including PDFs.
* **`chromadb`**: An open-source vector database used to store and retrieve document embeddings.
* **`transformers`**: Used for loading and running transformer models, often for local LLMs or tokenization.
* **`accelerate`**: A Hugging Face library for easily running PyTorch models on various hardware setups.
* **`bitsandbytes`**: Used for 8-bit quantization to run larger models with less memory.
* **`python-dotenv`**: For managing environment variables (API keys).
* **`tqdm`**: For progress bars during long operations (e.g., chunking, embedding).

---

## ⚙️ Setup Guide

Follow these steps to set up and run the RAG system:

### Prerequisites

* **Python**: Ensure you have Python 3.11 or 3.12 installed.
* **Git**: For cloning the repository.
* **Poppler**: Required by `unstructured` for robust PDF text extraction.
    * **Windows**: Download from [Poppler for Windows releases](https://github.com/oschwartz10612/poppler-windows/releases/). Extract it and add the `bin` directory (e.g., `C:\Program Files\poppler\poppler-23.xx.0\Library\bin`) to your System PATH.
    * **macOS**: `brew install poppler`
    * **Linux (Debian/Ubuntu)**: `sudo apt-get install poppler-utils`
* **Tesseract OCR**: Required by `unstructured` for extracting text from images within PDFs.
    * **Windows**: Download an installer (e.g., from [UB Mannheim](https://digi.bib.uni-mannheim.de/tesseract/)). During installation, ensure it's added to PATH. If not, add the installation directory (e.g., `C:\Program Files\Tesseract-OCR\`) to your System PATH.
    * **macOS**: `brew install tesseract`
    * **Linux (Debian/Ubuntu)**: `sudo apt-get install tesseract-ocr`

### 1. Clone the Repository

```bash
git clone https://github.com/Schrodingerscat00000/RAG-System-for-PDF.git
cd RAG-System-for-PDF
2. Create and Activate Virtual Environment
It's highly recommended to use a virtual environment.

Bash

# Create
python3.11 -m venv venv  # Use python3.12 if preferred

# Activate
# Windows (Command Prompt):
.\venv\Scripts\activate.bat
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate
3. Install Dependencies
Once the virtual environment is activated, install the required packages from requirements.txt:

Bash

pip install -r requirements.txt
4. Set Up API Keys
Create a .env file in the root directory of your project to store your API keys.

# .env file
COHERE_API_KEY="YOUR_COHERE_API_KEY"
# Optionally, if using Ollama locally:
# OLLAMA_BASE_URL="http://localhost:11434"
# OLLAMA_MODEL_NAME="qwen3:4b" # Make sure to pull this model: ollama pull qwen3:4b
Important: Replace "YOUR_COHERE_API_KEY" with your actual Cohere API Key. Cohere API key is used for both the LLM and the embedding model.

5. Prepare PDF Document
Place your HSC26-Bangla1st-Paper.pdf file inside a data/ directory in the root of your project:

your-repo-name/
├── data/
│   └── HSC26-Bangla1st-Paper.pdf
├── src/
│   ├── main.py
│   ├── config.py
│   ├── llm_model.py
│   ├── embedding_model.py
│   └── rag_system.py
├── .env
├── requirements.txt
└── README.md
6. Run the Application
Execute the main.py script from the src directory:

Bash

python -m src.main
The system will initialize (process PDF, create chunks, set up ChromaDB, load models). Once ready, you'll see a prompt: --- RAG System Ready! Start asking questions. ---.

💬 Sample Queries and Outputs
Here are some sample questions (in Bengali) and the corresponding answers generated by the system, demonstrating its multilingual RAG capabilities:

User Question: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
System Answer:
বিয়ের সময় কল্যাণীর প্রকৃত বয়স পনেরো বছর ছিল।

User Question: কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
System Answer:
অনুপমের ভাগ্য দেবতা হিসেবে তার মামাকে উল্লেখ করা হয়েছে। গল্পে এই বিষয়টি স্পষ্টভাবে উল্লেখ করা হয়েছে:   

"মামাকক ভাগয যদব্তািি প্রধান একিন্ট ব্লাি কািণ, তাি-"

এখানে, মামাকে অনুপমের ভাগ্য দেবতা হিসেবে উল্লে খ করা হয়েছে, যার কারণে অনুপমের জীবনে বিভিন্ন ঘটনা ঘটে।

User question: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?

Answer: অনুপমের ভাষায় "সুপুরুষ" কাকে বলা হয়েছে, তা স্পষ্টভাবে উল্লেখ নেই। তবে, প্রদত্ত সংগ্রহের ভিত্তিতে, অনুপমে ের মামাকে একজন হীনতা, লোভ ও অমানবিকতার প্রতীক হিসেবে চিত্রিত করা হয়েছে। অনুপমের মামা বিয়েতে নগদ টাকা ও গহনা পণ হি িসেবে দাবি করেন, যা তার হীনতা ও লোভকে প্রকাশ করে। এই সংগ্রহের ভিত্তিতে, "সুপুরুষ" শব্দটি অনুপমের মামার বিপরীত চরিত্ রকে নির্দেশ করতে পারে, যেমন অনুপমের পিতা বা অন্য কোনো সৎকর্মপরায়ণ ব্যক্তি। তবে, এটি স্পষ্টভাবে উল্লেখ না থাকলে নি িশ্চিত করা যাবে না।

📄 Answers to Assessment Questions
1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?
We used the unstructured library for text extraction from the PDF.

Why unstructured? It's a highly versatile and robust library designed for extracting clean text from various document types, including complex PDFs, scanned documents, and HTML. It intelligently handles different layouts, tables, and embedded images, aiming to preserve document structure.

Formatting Challenges: Yes, we faced challenges.

unstructured often relies on external tools like Poppler (for native PDF text extraction) and Tesseract OCR (for OCR on image-based text within PDFs). Without these installed and correctly added to the system's PATH, unstructured falls back to simpler, less accurate text extraction, potentially losing structure and quality, especially for scanned portions or complex layouts.

Initial runs showed warnings about Poppler and later Tesseract, necessitating their manual installation and PATH configuration for optimal extraction quality.

2. What chunking strategy did you choose (e.g., paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?
We implemented a Parent-Child Chunking (Hybrid) strategy using character limits:

Parent Chunks: Larger chunks (size 1000 characters) are created to provide broader context.

Child Chunks: Smaller, overlapping chunks (size 200 characters with 20 overlap) are created from the parent chunks. These smaller chunks are used for embedding and retrieval.

Why it works well for semantic retrieval:

Improved Search Accuracy: Smaller child chunks are more semantically focused. When a query is embedded, it can find a highly relevant small chunk more precisely than a large, diffuse one. This improves the accuracy of the initial semantic search.

Preserved Context: While child chunks are good for search, they might lack sufficient context for the LLM to generate a comprehensive answer. Once relevant child chunks are retrieved, their corresponding larger parent chunks are fetched. This provides the LLM with enough surrounding information to generate a grounded and complete response, addressing the "lost context" problem common with very small chunks.

Reduced Noise: By searching on smaller, more precise units, the retriever minimizes irrelevant information being included in the initial search results.

3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?
We initially attempted to use Voyage AI embeddings, then switched to Cohere Embeddings (specifically embed-multilingual-v3.0, which resolved to embed-v4.0 in practice) due to billing constraints. However, the final embedding model used in this system is bge-m3.

Why bge-m3?

Multilingual Support: bge-m3 (BGE-M3) is a powerful, state-of-the-art embedding model specifically designed for multilingual text. This is crucial for our system, which needs to handle both English and Bengali queries and process a Bengali PDF document effectively.

Strong Performance: BGE models, including bge-m3, are known for their high performance in various semantic retrieval tasks, often outperforming many proprietary and open-source alternatives.

Versatility: It supports various features like multi-granularity (word, sentence, document embeddings), which makes it highly adaptable for different chunking strategies and retrieval needs.

Local/Open-Source Friendly: As a Hugging Face model, bge-m3 can be run locally (though it requires significant resources) or via accessible APIs, offering flexibility in deployment.

How it Captures Meaning:

Like other embedding models, bge-m3 transforms human language (text) into high-dimensional numerical vectors. These vectors are constructed in such a way that texts with similar meanings or contexts are positioned closer to each other in this multi-dimensional vector space.

The model learns these semantic relationships by being trained on vast amounts of text data across multiple languages. This allows it to understand the nuances of language and generate unique numerical fingerprints for each piece of text.

By calculating the distance or similarity (e.g., cosine similarity) between these vectors, we can mathematically determine how semantically related any two pieces of text (like a user query and a document chunk) are. The closer the vectors are, the more semantically similar the texts.

4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?
Similarity Method: We use Cosine Similarity to compare the query embedding with the stored child chunk embeddings.

Why Cosine Similarity? It is the most common and effective method for comparing text embeddings. It measures the cosine of the angle between two vectors, indicating their directional similarity. A higher cosine similarity score (closer to 1) means the vectors are more semantically similar.

Storage Setup: We chose ChromaDB as our vector database.

Why ChromaDB?

Ease of Use: ChromaDB is lightweight and easy to set up for local development, requiring minimal configuration. It can run in-memory or persist to a local directory (chroma_db/).

Integration with LangChain: It has excellent native integration with LangChain, simplifying the process of adding, retrieving, and managing embeddings.

Local Persistence: Allows the vector store to be loaded from disk on subsequent runs, avoiding re-embedding the entire document every time the application starts.

5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?
Ensuring Meaningful Comparison:

High-Quality Embeddings: Using a multilingual, performant embedding model like bge-m3 ensures that both the user's query and the document chunks are transformed into semantically rich vector representations.

Appropriate Chunking: The Parent-Child chunking strategy ensures that small, semantically focused "child" chunks are used for initial retrieval, allowing for precise matching.

Cosine Similarity: This metric effectively measures the semantic proximity between the query and chunk embeddings.

Language Consistency: Since the embedding model is multilingual, it can effectively compare Bengali queries with Bengali document chunks.

What happens if the query is vague or missing context?

Retrieval of Less Relevant Chunks: A vague query will have a less precise semantic vector. This can lead to retrieving chunks that are broadly related but not directly relevant or might pull in too many diverse topics, diluting the quality of the context.

LLM Hallucination/Poor Grounding: With less relevant or insufficient context, the LLM might struggle to generate a truly grounded answer. It might resort to providing generic information, making assumptions, or even "hallucinating" facts that are not present in the retrieved documents, leading to an inaccurate or misleading response.

Incomplete Answers: If critical context is missed due to a vague query, the LLM's answer might be incomplete or lack specific details from the document.

6. Do the results seem relevant? If not, what might improve them?
Based on the provided sample queries, the results seem reasonably relevant. For "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?" and "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", the system successfully extracts correct facts directly from the document. For "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", while it states it's not explicitly mentioned, it attempts to infer context from the related retrieved information about the uncle's character, which shows some contextual understanding.

Potential Improvements:

Enhanced PDF Extraction: Fully resolving Poppler and Tesseract issues will ensure all text, including from images or complex layouts, is accurately extracted, leading to a more complete knowledge base.

Chunking Optimization:

Experimentation: Fine-tuning parent and child chunk sizes and overlap could further optimize retrieval.

Semantic Chunking: Implementing more advanced chunking strategies that use NLP techniques to create semantically coherent chunks rather than just character limits.

Retrieval Augmentation:

Re-ranking: After initial retrieval, use a re-ranking model (e.g., Cohere Rerank) to score the relevance of retrieved chunks more accurately, ensuring the very best chunks are passed to the LLM.

Hybrid Search: Combining vector search (semantic) with keyword search (lexical) can improve recall for specific terms.

Increasing k: Retrieving a slightly larger number of chunks (k) and then re-ranking them can increase the chance of getting highly relevant context.

LLM Prompt Engineering:

More Specific System Prompts: Guide the LLM with more explicit instructions on how to synthesize information from retrieved documents, prioritize grounded facts, and avoid conversational filler or explicit references to "retrieved documents" in the final answer.

Query Expansion/Rewriting: For vague queries, augment the user's query by expanding it with synonyms or by having a small LLM rewrite it for better search.

Evaluation Metrics: Implementing quantitative RAG evaluation frameworks (e.g., RAGAS) to measure aspects like groundedness, relevance, and answer faithfulness, allowing for systematic tracking of improvements.

Error Handling and User Feedback: More robust error handling and mechanisms for users to provide feedback on answer quality can help identify areas for improvement.
