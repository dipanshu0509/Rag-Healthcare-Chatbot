import gradio as gr
import requests

BACKEND_URL = "http://127.0.0.1:8000/chat"

def medical_chat(question):
    if not question or not question.strip():
        return "Please enter a medical question."

    try:
        response = requests.post(
            BACKEND_URL,
            json={"question": question},
            timeout=60
        )

        if response.status_code != 200:
            return f"Server error: {response.status_code}"

        data = response.json()

        answer = data.get("answer", "No answer returned.")
        disclaimer = data.get(
            "disclaimer",
            "Educational only. Not medical advice."
        )

        return f"{answer}\n\n⚠️ {disclaimer}"

    except requests.exceptions.ConnectionError:
        return "❌ Backend not running. Please start FastAPI server."
    except Exception as e:
        return f"Unexpected error: {e}"

gr.Interface(
    fn=medical_chat,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask a medical question (educational only)"
    ),
    outputs=gr.Textbox(lines=8),
    title="🩺 Medical Information Retrieval System",
    description=(
        "This system retrieves answers from indexed medical documents "
        "(WHO, NHM, guidelines).\n\n"
        "⚠️ Educational use only. Not a substitute for professional medical advice."
    )
).launch(share=True)
