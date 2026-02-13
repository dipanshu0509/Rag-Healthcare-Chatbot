from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
print("Loaded key:", os.getenv("GROQ_API_KEY"))


app = FastAPI(title="Medical Information Retrieval System (Groq)")

# ---------- Load FAISS ----------
embeddings = None

def get_embeddings():
    global embeddings
    if embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return embeddings

vectorstore = FAISS.load_local(
    "vectorstore/medical_knowledge",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ---------- Groq LLM ----------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# ---------- Prompt ----------
prompt = ChatPromptTemplate.from_template(
    """
You are a medical information assistant.

RULES:
- Use ONLY the provided context
- If the answer is not in the context, say you do not have enough information
- Do NOT give diagnosis or treatment advice
- Keep answers factual and concise

Context:
{context}

Question:
{question}

Answer (educational only):
"""
)

# ---------- API Schema ----------
class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    docs = retriever.invoke(query.question)

    if not docs:
        return {
            "answer": "I do not have enough information in the knowledge base to answer this question.",
            "disclaimer": "Educational only. Not medical advice."
        }

    context = "\n\n".join(doc.page_content for doc in docs)

    response = llm.invoke(
        prompt.format(
            context=context,
            question=query.question
        )
    )

    return {
        "answer": response.content.strip(),
        "disclaimer": "Educational only. Not medical advice."
    }
