from ingest.ingest import load_medalpaca
from ingest.pdf_ingest import load_pdfs

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

print("Loading sources...")

documents = []

# Source 1: MedAlpaca (optional – keep or remove)
documents += load_medalpaca(limit=1000)

# Source 2: PDFs
pdf_dir = "data/raw/pdfs"

if os.path.exists(pdf_dir):
    documents += load_pdfs(pdf_dir)
else:
    print("⚠️ PDF directory not found, skipping PDFs")

print("Total raw documents:", len(documents))

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

chunked_docs = []

for doc in documents:
    for chunk in splitter.split_text(doc.page_content):
        chunked_docs.append(
            Document(
                page_content=chunk,
                metadata=doc.metadata
            )
        )

print("Total chunks:", len(chunked_docs))

# Vector DB
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunked_docs, embeddings)
vectorstore.save_local("vectorstore/medical_knowledge")

print("✅ Knowledge base updated with PDFs")
