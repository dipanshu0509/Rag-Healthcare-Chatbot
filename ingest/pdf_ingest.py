import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

def load_pdfs(pdf_dir):
    documents = []

    for filename in os.listdir(pdf_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(pdf_dir, filename)
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        for page in pages:
            documents.append(
                Document(
                    page_content=page.page_content,
                    metadata={
                        "source": filename,
                        "type": "pdf"
                    }
                )
            )

    return documents
