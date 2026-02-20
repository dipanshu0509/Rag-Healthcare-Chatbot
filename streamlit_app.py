import streamlit as st
import uuid
import os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate



st.set_page_config(
    page_title="RAGMed Healthcare Assistant",
    page_icon="🩺",
    layout="centered"
)


# CSS

st.markdown("""
<style>

/* Page background */
html, body, .stApp {
    background-color: #f3f4f6;
}

/* Center white container */
.block-container {
    max-width: 450px;
    background-color: #04396b;
    padding: 25px;
    border-radius: 15px;
    margin-top: 40px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
}

/* Heading */
.heading {
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    color: #e1e7fa;
    margin-bottom: 5px;
}

.subheading {
    text-align: center;
    font-size: 18px;
    color: #e1e7fa;
    margin-bottom: 20px;
}

.warning {
    text-align: center;
    font-size: 15px;
    color: #F59E0B;   /* medical warning amber */
    margin-top: 8px;
}            

/* USER bubble */
.stChatMessage[data-testid="stChatMessage-user"] {
    background-color: #DCFCE7 !important;
    border-radius: 10px;
    padding: 10px;
}

/* USER text */
.stChatMessage[data-testid="stChatMessage-user"] * {
    color: #111827 !important;
}

/* ASSISTANT bubble */
.stChatMessage[data-testid="stChatMessage-assistant"] {
    background-color: #E0F2FE !important;
    border-radius: 10px;
    padding: 10px;
}

/* ASSISTANT text — IMPORTANT FIX */
.stChatMessage[data-testid="stChatMessage-assistant"] * {
    color: #010103 !important;
}

/* Input */
[data-testid="stChatInput"] {
    border-radius: 20px !important;
}

</style>
""", unsafe_allow_html=True)

# HEADING

st.markdown("""
<div class="heading">🩺 RAGMed Healthcare Assistant</div>
<div class="subheading">AI Powered Clinical Knowledge Support</div>
<div class="warning">⚠ For educational purposes only. Not for medical advice.</div>

""", unsafe_allow_html=True)


# LOAD ENV API Key

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY missing")
    st.stop()

# LOAD MODELS

@st.cache_resource
def load_models():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "vectorstore/medical_knowledge",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever()

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=api_key
    )

    return retriever, llm


retriever, llm = load_models()


prompt = ChatPromptTemplate.from_template("""
You are a medical assistant.

Use only given context.
Do not give medical advice.

Context:
{context}

Question:
{question}

Answer:
""")


# SESSION

if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# INPUT

user_input = st.chat_input("Type your medical question...")


if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    docs = retriever.invoke(user_input)

    if docs:

        context = "\n\n".join(d.page_content for d in docs)

        response = llm.invoke(
            prompt.format(
                context=context,
                question=user_input
            )
        )

        answer = response.content

    else:
        answer = "No information found."

    answer += "\n\n⚠ Educational only. Not medical advice."

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


# CHAT BUTTON

if st.button("New Chat"):
    st.session_state.messages = []
    st.rerun()