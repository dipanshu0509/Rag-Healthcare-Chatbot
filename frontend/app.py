import gradio as gr
import requests
import uuid

BACKEND_URL = "http://127.0.0.1:8000/chat"

# Create session ID
SESSION_ID = str(uuid.uuid4())

# Store full conversation text
conversation_history = ""


def medical_chat(question):
    global SESSION_ID
    global conversation_history

    if not question or not question.strip():
        return conversation_history + "\nPlease enter a medical question."

    try:
        response = requests.post(
            BACKEND_URL,
            json={
                "session_id": SESSION_ID,
                "question": question
            },
            timeout=60
        )

        if response.status_code != 200:
            answer = f"Server error: {response.status_code}"
        else:
            data = response.json()
            answer = data.get("answer", "No answer returned.")
            disclaimer = data.get(
                "disclaimer",
                "Educational only. Not medical advice."
            )
            answer = f"{answer}\n{disclaimer}"

    except requests.exceptions.ConnectionError:
        answer = "Backend not running. Please start FastAPI server."
    except Exception as e:
        answer = f"Unexpected error: {e}"

    # Append the conversation
    conversation_history += f"\n\n You: {question}\n\n RAGMed: {answer}"

    return conversation_history


def clear_chat():
    global SESSION_ID
    global conversation_history

    SESSION_ID = str(uuid.uuid4())  # Reset the backend session
    conversation_history = ""       # Clear frontend history

    return ""


#Custom Styling
custom_css = """
body {
    background-color: #f4f8fb;
}

.gradio-container {
    max-width: 850px !important;
    margin: auto;
}

#title {
    text-align: center;
    color: #0b5394;
}

#submit-btn {
    background-color: #0b5394 !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: bold;
}

#submit-btn:hover {
    background-color: #073763 !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        <h1 id="title">🩺 RAGMed: AI-Powered Healthcare Assistant</h1>
        <p style="text-align:center;">
        AI-powered system retrieving answers from medical guideline documents.
        </p>
        <p style="text-align:center; color:red;">
        Educational Use Only — Not a Substitute for Professional Medical Advice
        </p>
        """
    )

    with gr.Column():
        question_input = gr.Textbox(
            lines=3,
            placeholder="Ask your medical question here...",
            label="Enter Your Question"
        )

        with gr.Row():
            submit_btn = gr.Button("Get Answer", elem_id="submit-btn")
            clear_btn = gr.Button("New Chat")

        answer_output = gr.Textbox(
            lines=15,
            label="Medical Conversation",
            interactive=False
        )

    submit_btn.click(
        fn=medical_chat,
        inputs=question_input,
        outputs=answer_output
    )

    clear_btn.click(
        fn=clear_chat,
        outputs=answer_output
    )

demo.launch(share=True)
