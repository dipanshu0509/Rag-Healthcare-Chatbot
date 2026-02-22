# 🩺 RAGMed - AI Powered Conversational Healthcare Assistant

RAGMed is a production-style **Retrieval-Augmented Generation (RAG)** healthcare assistant that answers medical questions using trusted medical documents such as **WHO**, **NHM**, and clinical guideline PDFs.

The system retrieves relevant medical content from indexed documents using a **FAISS vector database** and generates accurate, context-aware responses using **Groq LLM**, without relying on hardcoded answers.

The project supports **Streamlit frontend (primary deployed UI)**, **FastAPI backend**, and an optional **Gradio frontend**, following a modular production-ready architecture.

---

## 🚀 Features

- Multi-source ingestion (PDF + medical datasets)
- FAISS vector database for fast semantic similarity search
- Retrieval-Augmented Generation (RAG) pipeline
- Context-aware responses using Groq LLM
- FastAPI production-ready backend
- Streamlit interactive frontend (primary deployed interface)
- Optional Gradio frontend support
- Modular and scalable architecture
- Uses trusted medical knowledge sources
- No hardcoded responses

---

## 🏗 Architecture

```
User → Gradio UI → FastAPI → FAISS Retrieval → Groq LLM → Response
```


---

## 🔄 Workflow

1. User submits a medical query via Streamlit or Gradio UI  
2. FastAPI backend receives and processes the request  
3. Query is converted into embeddings  
4. FAISS retrieves relevant medical document chunks  
5. Retrieved context is passed to Groq LLM  
6. Groq LLM generates contextual medical response  
7. Response is displayed in the frontend  

---

## 📚 Data Sources

- WHO Guidelines  
- NHM Documents  
- Clinical Medical PDFs  
- Structured medical datasets  

---

## 🛠 Tech Stack

### Language
Python 3.10.19

### Backend
- FastAPI  
- LangChain  
- FAISS  
- Groq LLM  

### Frontend
- Streamlit (Primary deployed interface)  
- Gradio (Optional interface)  

### Embeddings
- HuggingFace MiniLM Embeddings  

### Other Tools
- dotenv  
- Uvicorn  

---

# ⚙️ Setup Instructions

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/dipanshu0509/Rag-Healthcare-Chatbot/blob/main/Readme.md
cd <your project path>
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Add Environment Variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

# ▶️ Running the Application

## 🔹 Run Streamlit Frontend (Recommended)
```bash
streamlit run streamlit_app.py
```

Open in browser:
```bash
http://localhost:8501
```

## 🔹 Step 1: Run FastAPI Backend

```bash
uvicorn app:app --reload
```

Default backend URL:

```
http://127.0.0.1:8000
```

---

## 🔹 Step 2: Run Gradio Frontend

In a new terminal:

```bash
python app.py
```

If using share mode:

```python
demo.launch(share=True)
```

Gradio URL will be generated in terminal.

---

# 🗂 Project Structure (Example)

```
medical-rag/
├── Frontend/
      ├── app.py
├── data/raw/pdfs
├── ingest
      ├── pdf_ingest.py      
├── app.py
├── build_knowledge_base.py
├── streamlit_app.py
├── 26087.jpg
├── vectorstore/medical_knowledge
├── requirements.txt
├── .env
└── README.md
```

---

# ⚠️ Disclaimer

This system is built for **educational and research purposes only**.
It does **not** replace professional medical consultation.

Always consult a qualified healthcare provider for medical advice.
