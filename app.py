from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
from dotenv import load_dotenv
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# Loading Environment Variables(api key)
load_dotenv(dotenv_path=".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")


# Initializing FastAPI
app = FastAPI(title="RAGMed - Conversational Medical RAG System")


# Loading the FAISS VectorDB Store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vectorstore/medical_knowledge",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# Initializing the Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY
)


# Prompt to retrieve answer
prompt = ChatPromptTemplate.from_template(
    """
You are a medical information assistant.

RULES:
- Use ONLY the provided context
- If the answer is not in the context, say you do not have enough information
- Do NOT give diagnosis or treatment advice
- Keep answers factual and concise

Conversation History:
{history}

Context:
{context}

Current Question:
{question}

Answer (educational only):
"""
)


# In Memory Session Storage
chat_sessions: Dict[str, List[Dict[str, str]]] = {}


# Request Schema
class Query(BaseModel):
    session_id: str
    question: str



# Chat Endpoint
@app.post("/chat")
def chat(query: Query):

    session_id = query.session_id

    # Creating session if not exists
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    history = chat_sessions[session_id]

    # Format history for prompt
    formatted_history = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in history]
    )

    # To retrieve relevant documents
    docs = retriever.invoke(query.question)

    if not docs:
        return {
            "answer": "I do not have enough information in the knowledge base to answer this question.",
            "disclaimer": "Educational only. Not medical advice."
        }

    context = "\n\n".join(doc.page_content for doc in docs)

    # Generating LLM response
    response = llm.invoke(
        prompt.format(
            history=formatted_history,
            context=context,
            question=query.question
        )
    )

    answer = response.content.strip()

    # Save conversation history
    chat_sessions[session_id].append({
        "role": "user",
        "content": query.question
    })

    chat_sessions[session_id].append({
        "role": "assistant",
        "content": answer
    })

    # Limit memory (keeping the last 5 conversations = 10 messages)
    if len(chat_sessions[session_id]) > 10:
        chat_sessions[session_id] = chat_sessions[session_id][-10:]

    return {
        "answer": answer,
        "disclaimer": "Educational only. Not medical advice."
    }
