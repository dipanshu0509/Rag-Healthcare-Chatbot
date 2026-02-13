from datasets import load_dataset
from langchain_core.documents import Document

def load_medalpaca(limit=1000):
    dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")

    documents = []

    for i in range(limit):
        row = dataset["train"][i]

        question = row["input"].strip()
        answer = row["output"].strip()

        text = f"Question: {question}\nAnswer: {answer}"

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": "MedAlpaca",
                    "type": "medical_qa"
                }
            )
        )

    return documents
