import streamlit as st
import os
import base64
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# page configuration 

st.set_page_config(
    page_title="RAGMed-AI Healthcare Assistant",
    page_icon="🩺",
    layout="wide"
)
# image

def get_base64_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

img_base64 = get_base64_image("26087.jpg")


# CSS

st.markdown(f"""
<style>

.stApp {{
    background-image: url("data:image/jpeg;base64,{img_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stAppViewContainer"] {{
    background: transparent;
}}

[data-testid="stHeader"] {{
    background: transparent;
}}

.block-container {{
    max-width: 100% !important;
    padding-top: 140px;
    padding-bottom: 120px;
    padding-left: 12%;
    padding-right: 12%;
}}

.fixed-header {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    background-color: transparent;
    padding: 18px;
    text-align: center;
    z-index: 1000;
    box-shadow: none;
}}

.heading {{
    font-size: 40px;
    font-weight: bold;
    color: white;
}}

.subheading {{
    font-size: 20px;
    color: #cbd5e1;
}}

.warning {{
    font-size: 17px;
    color: #f59e0b;
}}

.chat-container {{
    max-width: 850px;
    margin: auto;
}}

.chat-wrapper {{
    max-width: 700px;
    margin: auto;
    padding-left: 20px;
    padding-right: 20px;
}}

.user-box {{
    background: rgba(255, 255, 255, 0.75);
    color: #000000;
    padding: 14px 18px;
    border-radius: 18px;
    margin: 12px 0px;
    margin-left: auto;
    font-size: 17px;
    font-family: "Inter", "Segoe UI", sans-serif;
    line-height: 1.5;
    border: 1px solid #c2a46d;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.25);
    width: fit-content;
    max-width: 65%; 
    backdrop-filter: blur(6px);  
    -webkit-backdrop-filter: blur(6px);
}}


.bot-box {{
    background: rgba(255, 255, 255, 0.75);
    color: #000000;
    padding: 16px 20px;
    border-radius: 18px;
    margin: 12px 0px;
    margin-right: auto;
    font-size: 17px;
    font-family: "Inter", "Segoe UI", sans-serif;
    line-height: 1.5;
    border: 1px solid #c2a46d;
    box-shadow: 2px 2px 12px rgba(0,0,0,0.30);
    width: fit-content;
    max-width: 65%;              
    backdrop-filter: blur(6px);  
    -webkit-backdrop-filter: blur(6px);
}}

div[data-testid="stChatInput"] {{
    position: fixed !important;
    bottom: 20px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: 700px !important;
    z-index: 1000 !important;
}}

div[data-testid="stButton"] > button {{
    position: fixed !important;
    bottom: 27px !important;
    left: calc(42% - 380px) !important;
    width: 110px !important;
    height: 45px !important;
    border-radius: 10px !important;
    z-index: 1001 !important;
}}

div[data-testid="stButton"] {{
    background: transparent !important;
}}

</style>
""", unsafe_allow_html=True)

# header
st.markdown("""
<div class="fixed-header">
    <div class="heading">🩺 RAGMed - AI Healthcare Assistant</div>
    <div class="subheading">AI Powered Clinical Knowledge Support</div>
    <div class="warning">⚠ For educational purposes only. Not intended as medical advice.</div>
</div>
""", unsafe_allow_html=True)


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY missing")
    st.stop()

#Models loading

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
You are an AI-powered medical knowledge assistant.

STRICT RULES:

1. Use ONLY the information provided in the context.
2. Do NOT use your own knowledge.
3. If the answer is not in the context, say:
   "I cannot find this information in the provided medical documents."

4. Do NOT provide medical advice, diagnosis, treatment, prescription, or procedures.
5. Do NOT simulate being a doctor, nurse, or emergency responder.
6. If the user describes a medical emergency (examples: chest pain, breathing difficulty, heart attack symptoms, unconsciousness, severe bleeding), respond ONLY with:

   "This may be a medical emergency. Please seek immediate medical attention or contact emergency services or a qualified healthcare professional."

7. Do NOT provide step by step medical instructions.
8. Keep responses informational and educational only.
9. Do not hallucinate or invent information.
10. Always prioritize safety.
11. Always mention "Educational purposes only. Please consult a doctor before making any medical decisions." in every treatment or medicine advice answer.

Context:
{context}

Question:
{question}

Answer:
""")



if "messages" not in st.session_state:
    st.session_state.messages = []


#chat area
st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-box">🧑‍💻 {msg["content"]}</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f'<div class="bot-box">🩺 {msg["content"]}</div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="bottom-bar"><div class="bottom-container">', unsafe_allow_html=True)

col1, col2 = st.columns([1,6])

with col1:
    if st.button("New Chat", key="new_chat_button_fixed"):
        st.session_state.messages = []
        st.rerun()

with col2:
    user_input = st.chat_input(
        "Type your medical question...",
        key="chat_input_fixed"
    )

st.markdown('</div></div>', unsafe_allow_html=True)


# user input
if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    st.markdown(
        f'<div class="user-box">🧑‍💻 {user_input}</div>',
        unsafe_allow_html=True
    )

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

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()
