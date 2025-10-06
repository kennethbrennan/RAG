# RAG Document Analyzer & Interactive Chat

This application is a Retrieval-Augmented Generation (RAG) system designed to parse, classify, and index technical documentation (PDFs) to enable high-quality, factual question-answering via an interactive command-line interface.

It leverages a Zero-Shot Classifier for smart chunking and topic-based indexing, ensuring retrieval is contextually accurate and relevant.

---

## Features
- PDF Parsing with Page Metadata: Extracts text from PDFs and preserves page numbers for accurate source attribution.  
- Intelligent Chunking: Splits text into semantically relevant chunks for vector indexing.  
- Zero-Shot Classification: Categorizes chunks into predefined topics (e.g., Scope_of_Work, Requirements) for better retrieval.  
- Targeted Vector Indexing: Stores chunks in multiple vector store collections (e.g., Chroma) based on classification.  
- Local LLM Integration: Uses qwen3:8b via Ollama for synthesis.  
- Automatic Device Fallback: Detects CUDA GPU and defaults to CPU if unavailable.  
- Stateless Interactive Chat: Command-line interface where each query is independent (no memory across turns).  

---

## Prerequisites
- Python 3.x  
- Ollama running locally  
- Model pulled via Ollama (ollama pull qwen3:8b)  
- Project structure with custom classes:
  classes/
    util.py
    pdf_parser.py
    llm.py
    ...

---

## Setup and Installation

Clone the repository:
```bash
git clone https://github.com/kennethbrennan/rag.git
cd rag
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Download model with Ollama:
```bash
ollama pull qwen3:8b
```
---

## Configuration

Update configuration constants at the top of your main script:

Constant                | Description                                     | Action Required
------------------------|-------------------------------------------------|----------------------------------
model_name              | Ollama model for RAG (e.g., qwen3:8b).          | Confirm model is downloaded.
vector_store_local_path | Local directory for vector index storage.        | Update to valid local path.
pdf_location            | Path to the PDF you want to process.             | Update to your document path.
categories              | Topic list for Zero-Shot Classifier.             | Adjust as needed.

---

## How to Run

Run the main script (e.g., main_rag.py):
```bash
python rag.py
```

Indexing Phase: The script will parse the PDF, chunk text, classify, and add to vector stores.  
Interactive Chat Phase: Starts an interactive CLI. Type your queries and press Enter.  

