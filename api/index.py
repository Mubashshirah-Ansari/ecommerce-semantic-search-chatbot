import re
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()


# Load the same dataset used in the case study
DATASET_URL = (
    "https://huggingface.co/datasets/bitext/"
    "Bitext-customer-support-llm-chatbot-training-dataset/"
    "resolve/main/"
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)

df = pd.read_csv(DATASET_URL)


# Clean customer questions
df["clean_instruction"] = df["instruction"].apply(
    lambda x: re.sub(r"\{\{.*?\}\}", "", x)
)

df["clean_instruction"] = (
    df["clean_instruction"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
    .str.lower()
)


# Create TF-IDF vectors
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    lowercase=True
)

train_vectors = vectorizer.fit_transform(
    df["clean_instruction"]
)


def semantic_search(query):

    # Clean the user query
    clean_query = re.sub(r"\{\{.*?\}\}", "", query)
    clean_query = re.sub(r"\s+", " ", clean_query).strip().lower()

    # Convert query into TF-IDF vector
    query_vector = vectorizer.transform([clean_query])

    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_vector,
        train_vectors
    )[0]

    # Find best matching question
    best_index = similarities.argmax()

    matched_question = df.iloc[best_index]["instruction"]
    intent = df.iloc[best_index]["intent"]
    response = df.iloc[best_index]["response"]
    score = float(similarities[best_index])

    return matched_question, intent, score, response


def extract_order_number(query):

    pattern = r"\b\d{4,}\b"
    match = re.search(pattern, query)

    if match:
        return match.group()

    return None


def fill_parameters(response, order_number):

    if order_number:
        response = response.replace(
            "{{Order Number}}",
            order_number
        )

    return response


def chatbot_search(query, threshold=0.20):

    matched_question, intent, score, response = semantic_search(query)

    if score < threshold:
        return {
            "intent": None,
            "similarity": score,
            "order_number": None,
            "response": (
                "Sorry, I could not understand your question. "
                "Please rephrase it."
            )
        }

    order_number = extract_order_number(query)

    response = fill_parameters(
        response,
        order_number
    )

    return {
        "intent": intent,
        "similarity": score,
        "order_number": order_number,
        "response": response
    }


class ChatRequest(BaseModel):
    query: str


@app.get("/api")
def home():

    return {
        "message": "E-commerce Semantic Search Chatbot API"
    }


@app.post("/api/chat")
def chat(request: ChatRequest):

    if not request.query.strip():
        return {
            "error": "Please enter a question."
        }

    return chatbot_search(request.query)
