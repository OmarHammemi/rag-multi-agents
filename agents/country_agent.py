import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import faiss
import json
import numpy as np
import re
from openai import OpenAI
from dotenv import load_dotenv
import os

# === Load env and OpenAI key ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# === Load FAISS index and metadata ===
country_index = faiss.read_index("/home/omar/rag/index/country_faiss.index")
with open("/home/omar/rag/data/country_metadata.json", "r", encoding="utf-8") as f:
    country_metadata = json.load(f)

# === Embedding function ===
def embed_query(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding).astype("float32").reshape(1, -1)

# === Extract structured fields from country info ===
def extract_country_summary(text: str) -> str:
    summary = {}

    if match := re.search(r"([A-Za-z\s]+)\s+is the capital of ([A-Za-z\s]+)", text):
        summary["capital"] = f"The capital of {match.group(2)} is {match.group(1)}."

    if match := re.search(r"total area of ([\d,]+)\s*square kilometers", text):
        summary["area"] = f"It covers an area of {match.group(1)} sq km."

    if match := re.search(r"population of ([\d,]+)", text):
        summary["population"] = f"It has a population of {match.group(1)}."

    if match := re.search(r"official languages? spoken.*?:?\s*([A-Za-z,\s]+)", text):
        summary["language"] = f"The official languages include {match.group(1).strip()}."

    if match := re.search(r"National Animal is the ([A-Za-z\s]+)", text):
        summary["animal"] = f"The national animal is the {match.group(1).strip()}."

    if match := re.search(r"National Bird is the ([A-Za-z\s]+)", text):
        summary["bird"] = f"The national bird is the {match.group(1).strip()}."

    # Limit cultural info to first sentence
    if match := re.search(r"### About the Country best play:(.+)", text, re.DOTALL):
        full = match.group(1).strip()
        first_sentence = re.split(r"\.|\n", full)[0]
        summary["extra"] = f"Notable cultural fact: {first_sentence.strip()}."

    return " ".join(summary.values())

# === Main country agent function ===
def country_agent(state: dict) -> dict:
    try:
        query = state["user_query"].lower()

        # Try to match a country name exactly
        matched = None
        for entry in country_metadata:
            if entry["id"].lower() in query:
                matched = entry
                break

        results = []

        if matched:
            summary = extract_country_summary(matched["text"])
            results.append(f"🌍 {matched['id']} (Exact Match) — {summary}")
        else:
            # Semantic fallback
            query_embedding = embed_query(query)
            distances, indices = country_index.search(query_embedding, k=3)

            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(country_metadata):
                    entry = country_metadata[idx]
                    summary = extract_country_summary(entry["text"])
                    results.append(f"🌍 **{entry['id']}** (Score: {dist:.2f}) — {summary}")

        state["answer"] = "\n\n---\n\n".join(results) if results else "❌ No matching country found."

    except Exception as e:
        state["answer"] = f"❌ Error in country_agent: {str(e)}"

    return state

# # === Test Example ===
if __name__ == "__main__":
    state = {"user_query": "What is the capital of Tunisia?"}
    result = country_agent(state)
    print(result["answer"])
