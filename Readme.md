# 🩺 Medical Information Retrieval System (RAG-Based)

A production-style **Retrieval-Augmented Generation (RAG)** system that answers medical questions using indexed medical documents such as **WHO**, **NHM**, and clinical guideline PDFs.

This system retrieves relevant medical content from trusted sources and generates responses using an LLM — **without hardcoded answers**.

---

## 🚀 Features

* ✅ Multi-source ingestion (PDF + datasets)
* ✅ FAISS vector database for fast similarity search
* ✅ Retrieval-based answering (No hallucinated hardcoded replies)
* ✅ Groq LLM integration
* ✅ FastAPI backend
* ✅ Gradio frontend UI
* ✅ Modular production-style architecture

---

## 🏗 Architecture

```
User → Gradio UI → FastAPI → FAISS Retrieval → Groq LLM → Response
```

### Flow Explanation:

1. User submits a medical query via Gradio UI
2. FastAPI backend receives the request
3. FAISS retrieves relevant medical document chunks
4. Retrieved context is passed to Groq LLM
5. LLM generates contextual response

---

## 📚 Data Sources

* WHO Guidelines
* NHM Documents
* Clinical PDFs
* Structured medical datasets

---

## 🛠 Tech Stack

* Python
* FastAPI
* FAISS
* LangChain
* Groq
* Gradio

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
│
├── backend.py
├── app.py
├── ingestion.py
├── vectorstore/
├── data/
├── requirements.txt
├── .env
└── README.md
```

---

# ⚠️ Disclaimer

This system is built for **educational and research purposes only**.
It does **not** replace professional medical consultation.

Always consult a qualified healthcare provider for medical advice.
