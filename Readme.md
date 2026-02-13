# 🩺 Medical Information Retrieval System (RAG-Based)

A production-style Retrieval-Augmented Generation (RAG) system that answers medical questions using indexed medical documents such as WHO, NHM, and guideline PDFs.

## 🚀 Features

- Multi-source ingestion (PDF + datasets)
- FAISS vector database
- Retrieval-based answering
- Groq LLM integration
- FastAPI backend
- Gradio frontend
- No hardcoded medical responses

## 🏗 Architecture

User → Gradio UI → FastAPI → FAISS Retrieval → Groq LLM → Response

## 📚 Data Sources

- WHO Guidelines
- NHM Documents
- Clinical PDFs
- Medical datasets

## ⚠️ Disclaimer

Educational purposes only. Not medical advice.

## 🛠 Tech Stack

- Python
- FastAPI
- FAISS
- LangChain
- Groq
- Gradio
