import streamlit as st
import uuid
import os
import base64
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="RAGMed Healthcare Assistant",
    page_icon="🩺",
    layout="centered"
)


# ---------------- BACKGROUND IMAGE FUNCTION ----------------

def get_base64_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

img_base64 = get_base64_image("WW.jpg")


# ---------------- CSS ----------------

st.markdown(f"""
<style>

/* Background image */
.stApp {{
    background-image: url("data:image/jpeg;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* Remove default white areas */
[data-testid="stAppViewContainer"] {{
    background: transparent;
}}

[data-testid="stHeader"] {{
    background: transparent;
}}

/* Main container card */
.block-container {{
    max-width: 450px;
    background-color: rgba(4, 57, 107, 0.95);
    padding: 25px;
    border-radius: 15px;

    margin-left: auto;
    margin-right:auto;
    margin-top: 10px;
    box-shadow: 0px 4px 25px rgba(0,0,0,0.5);
}}

/* Heading */
.heading {{
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    color: #e1e7fa;
    margin-bottom: 5px;
    width: 100%;
}}

.subheading {{
    text-align: center;
    font-size: 18px;
    color: #e1e7fa;
    margin-bottom: 15px;
    width: 100%;
}}

.warning {{
    text-align: center;
    font-size: 14px;
    color: #F59E0B;
    margin-bottom: 20px;
    width: 100%;
}}

/* User bubble */
.stChatMessage[data-testid="stChatMessage-user"] {{
    background-color: #DCFCE7 !important;
    border-radius: 10px;
    padding: 10px;
}}

.stChatMessage[data-testid="stChatMessage-user"] * {{
    color: #111827 !important;
}}

/* Assistant bubble */
.stChatMessage[data-testid="stChatMessage-assistant"] {{
    background-color: #E0F2FE !important;
    border-radius: 10px;
    padding: 10px;
}}

.stChatMessage[data-testid="stChatMessage-assistant"] * {{
    color: #000000 !important;
}}

/* Input box */
[data-testid="stChatInput"] {{
    border-radius: 20px !important;
}}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADING ----------------

st.markdown("""
<div class="heading">🩺 RAGMed Healthcare Assistant</div>
<div class="subheading">AI Powered Clinical Knowledge Support</div>
<div class="warning">⚠ For educational purposes only. Not medical advice.</div>
""", unsafe_allow_html=True)


# ---------------- LOAD ENV ----------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY missing")
    st.stop()


# ---------------- LOAD MODELS ----------------

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


# ---------------- PROMPT ----------------

prompt = ChatPromptTemplate.from_template("""
You are a medical assistant.

Use only the provided context.
Do not give medical advice.

Context:
{context}

Question:
{question}

Answer:
""")


# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- DISPLAY CHAT ----------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------------- USER INPUT ----------------

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
        answer = "No relevant medical information found."

    answer += "\n\n⚠ Educational purposes only. Not medical advice."

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


# ---------------- NEW CHAT BUTTON ----------------

if st.button("New Chat"):
    st.session_state.messages = []
    st.rerun()
