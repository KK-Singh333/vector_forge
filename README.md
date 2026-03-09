# Scalable FAISS Store & RAG System

A high-performance, scalable vector storage and retrieval system built using **FAISS**, **SQLite**, and **FastAPI**. This project is designed to power Retrieval-Augmented Generation (RAG) applications with features like automatic index optimization, segmented storage, and a dedicated LLM chat interface.

## Features

- **Scalable Vector Management**: Custom `VectorStoreManager` that handles FAISS indices, supporting automatic upgrades from Flat (exact) to IVF (approximate) indices as data volume grows.
- **Segmented Storage**: Efficiently manages vector data in rotated segments to handle large datasets.
- **Hybrid Metadata Store**: Uses SQLite to persist relationships between vector IDs, users, and document chunks (PDFs).
- **RAG Pipeline**: Integrated Chat Service that rewrites user queries for semantic search, retrieves context, and generates answers using LLMs (Groq).
- **Ingestion API**: Automated pipeline for PDF ingestion, parsing, chunking, and embedding.
- **Performance Evaluation**: Includes tools to benchmark retrieval latency and accuracy (Top-k hits).

## Architecture

The system is composed of two primary services:

1.  **Knowledge Service (Storage)-app/**
    - **Port**: `8068` 
    - **Responsibilities**: Manages the FAISS vector store and SQLite database. Handles the heavy lifting of document ingestion (`/ingest`) and vector similarity search (`/search`).
    - **Key Components**: `VectorStoreManager`, `DataBaseStoreManager`, `IngestionService`.

2.  **LLM Service (Chat)-llm_server/**
    - **Port**: `8069`
    - **Responsibilities**: Acts as the user-facing gateway. It handles query rewriting, communicates with the Knowledge Service for context, and interfaces with the Groq API for response generation.
    - **Key Components**: `ChatService`, `LLMClient`, `KnowledgeClient`.

## Prerequisites

- Python 3.8+
- FAISS (CPU or GPU version)
- Groq API Key (for LLM generation)

## Configuration

Ensure you have the necessary environment variables set, particularly for the LLM client:

```bash
export GROQ_API_KEY="your_groq_api_key"
```

## Usage

### 1. Starting the Services
Start vector server

```bash
cd app
python -m api.main
# Server will start on http://127.0.0.1:8068
```

Start the LLM/Chat Server:

```bash
cd llm_server
python main.py
# Server will start on http://127.0.0.1:8069
```

_(Note: Ensure the underlying app(vector database engine) server is running on port 8068 to handle vector operations)._

### 2. Register

Upload documents (PDFs) to the system via the API:

```http
POST /register
Content-Type: application/json
username: <Username you want>
```

### 3. Ingestion

Upload documents (PDFs) to the system via the API:

```http
POST /ingest
Content-Type: multipart/form-data

user_id: 1
pdf_id: 1
file: <your_file.pdf>
```

### 4. Chat Interface

Query the system to retrieve information from ingested documents:

```http
POST /chat
Content-Type: application/json

{
  "user_id": 1,
  "query": "What is the total area of the building?",
  "k": 5
}
```

### 5. Evaluation

Run the evaluation script to test retrieval performance against a ground-truth dataset:

```bash
python scripts/evaluate.py
```

**Output Metrics:**

-Total Queries:   20 

-Average Latency: 0.515 seconds

-P95 Latency:     0.701 seconds

-Top-1 Accuracy:  70.0%

-Top-3 Accuracy:  90.0%
