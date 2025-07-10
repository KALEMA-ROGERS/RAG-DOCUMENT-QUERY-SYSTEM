
# Intelligent System (RAG-based)

# Overview
This project allows users to chat with large documents using Retrieval-Augmented Generation (RAG), powered by Langchain, Groq LLM, and HuggingFace Embeddings

## Features
- Document ingestion, chunking, and embedding via HuggingFace Sentence Transformers
- Vector storage and semantic retrieval with ChromaDB
- Question answering using Groq API
- Streamlit web interface for seamless user interaction
- Dockerized for easy local development & production deployment

## Architecture
- Langchain (RAG pipeline)
- ChromaDB (vector store)
- Streamlit (frontend)
- Docker & Docker Compose (deployment)

## 1. Clone the Repository
```bash
git clone https://github.com/KALEMA-ROGERS/RAG-DOCUMENT-QUERY-SYSTEM.git
cd rag-document-query-system
```

## 3. Build Docker Containers
```bash
docker-compose build


## 4. Run the Application
```bash
docker-compose up
```
## Or Run it Locally
```
activate the virtual environment
venv\Scripts\activate
python -m streamlit run app/streamlit_app.py

### 5. Access
Open your browser at: http://localhost:8501



