import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import faiss
import json
import numpy as np
import re
from openai import OpenAI
from dotenv import load_dotenv

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
def extract_country_summary(text: str, country_name: str) -> str:
    summary = {}

    if match := re.search(r"([A-Za-z\s]+)\s+is the capital of ([A-Za-z\s]+)", text):
        capital = match.group(1).strip()
        country = match.group(2).strip()
        if country.lower() == country_name.lower():
            summary["capital"] = f"The capital of {country} is {capital}."

    if match := re.search(r"total area of ([\d,]+)\s*square kilometers", text):
        summary["area"] = f"It covers an area of {match.group(1)} sq km."

    if match := re.search(r"population of ([\d,]+)", text):
        summary["population"] = f"It has a population of {match.group(1)}."

    if match := re.search(r"official languages? spoken.*?:?\s*([A-Za-z,\s]+)", text):
        lang = match.group(1).strip()
        if lang:
            summary["language"] = f"The official languages include {lang}."

    if match := re.search(r"National Animal is the ([A-Za-z\s]+)", text):
        summary["animal"] = f"The national animal is the {match.group(1).strip()}."

    if match := re.search(r"National Bird is the ([A-Za-z\s]+)", text):
        summary["bird"] = f"The national bird is the {match.group(1).strip()}."

    if match := re.search(r"### About the Country best play:(.+)", text, re.DOTALL):
        full = match.group(1).strip()
        first_sentence = re.split(r"\\.|\\n", full)[0]
        if first_sentence:
            summary["extra"] = f"Notable cultural fact: {first_sentence.strip()}."

    return " ".join(summary.values()).strip()

# === Main country agent function ===
def country_agent(state: dict) -> dict:
    try:
        query = state["user_query"].lower()

        matched = None
        for entry in country_metadata:
            if entry["id"].lower() in query:
                matched = entry
                break

        if matched:
            summary = extract_country_summary(matched["text"], matched["id"])
            if summary:
                state["answer"] = f"🌍 {matched['id']} (Exact Match) — {summary}"
            else:
                state["answer"] = f"✅ Found {matched['id']}, but no useful info was available."
        else:
            query_embedding = embed_query(query)
            distances, indices = country_index.search(query_embedding, k=1)

            idx = indices[0][0]
            score = distances[0][0]
            if idx < len(country_metadata) and score < 0.25:
                entry = country_metadata[idx]
                summary = extract_country_summary(entry["text"], entry["id"])
                if summary:
                    state["answer"] = f"🌍 **{entry['id']}** (Score: {score:.2f}) — {summary}"
                else:
                    state["answer"] = "❌ No meaningful information found for the query."
            else:
                state["answer"] = "❌ No matching country found."

    except Exception as e:
        state["answer"] = f"❌ Error in country_agent: {str(e)}"

    return state

