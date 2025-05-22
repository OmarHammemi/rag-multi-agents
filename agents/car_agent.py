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
car_index = faiss.read_index("/home/omar/rag/index/car_faiss.index")
with open("/home/omar/rag/data/car_metadata.json", "r", encoding="utf-8") as f:
    car_metadata = json.load(f)

# === Embed a query ===
def embed_query(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding).astype("float32").reshape(1, -1)

# === Parse structured car specs ===
def parse_specs(text: str) -> dict:
    specs = {
        "body_type": None,
        "horsepower": None,
        "top_speed": None,
        "engine_size_cc": None,
        "mileage_km_l": None,
        "ncap": None
    }

    if match := re.search(r"Specs:\s*(.+)", text, re.IGNORECASE):
        parts = [p.strip().lower() for p in match.group(1).split(",")]
        if parts:
            specs["body_type"] = parts[0]
        for part in parts:
            if "km/h" in part and (speed := re.search(r"(\d+)\s*km/h", part)):
                specs["top_speed"] = int(speed.group(1))
            if "km/l" in part and (mil := re.search(r"(\d+)\s*km/l", part)):
                specs["mileage_km_l"] = int(mil.group(1))

    if match := re.search(r"Engine:\s*(.+)", text, re.IGNORECASE):
        line = match.group(1).lower()
        if hp := re.search(r"(\d+)\s*hp", line):
            specs["horsepower"] = int(hp.group(1))
        if cc := re.search(r"(\d+)\s*cc", line):
            specs["engine_size_cc"] = int(cc.group(1))

    if match := re.search(r"NCAP Rating:\s*([\d.]+)/5", text):
        specs["ncap"] = float(match.group(1))

    return specs

# === Extract brand name from text ===
def extract_brand(text: str) -> str:
    match = re.search(r"by\s+([^\(]+)\s*\(Launched", text)
    return match.group(1).strip() if match else "Unknown brand"

# === Extract description section ===
def extract_description(text: str) -> str:
    match = re.search(r"Description:(.+?)Engine:", text, re.DOTALL | re.IGNORECASE)
    return f"Description:{match.group(1).strip()}" if match else ""
# === Main car agent ===
def car_agent(state: dict) -> dict:
    try:
        query = state["user_query"].lower()
        query_embedding = embed_query(query)
        distances, indices = car_index.search(query_embedding, k=6)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(car_metadata):
                meta = car_metadata[idx]
                specs = parse_specs(meta["text"])
                results.append({
                    "id": meta["id"],
                    "text": meta["text"],
                    "score": float(dist),
                    **specs
                })
        # Dynamic score boost based on query content
        def rank(car):
            base = -car["score"]
            if "speed" in query or "fast" in query:
                base += 0.01 * (car.get("top_speed") or 0)
            if "efficient" in query or "mileage" in query:
                base += 0.01 * (car.get("mileage_km_l") or 0)
            if "safe" in query or "ncap" in query:
                base += 0.1 * (car.get("ncap") or 0)
            if "power" in query or "horsepower" in query:
                base += 0.01 * (car.get("horsepower") or 0)
            return base
        top = sorted(results, key=rank, reverse=True)[:3]
        lines = []
        for car in top:
            lines.append(
                f"🚗 Car: **{car['id']}** with a relevance score of {car['score']:.2f}. "
                f"This {car.get('body_type', 'unknown type')} was launched by {extract_brand(car['text'])}. "
                f"It features an engine with {car.get('horsepower', 'N/A')} HP, a top speed of {car.get('top_speed', 'N/A')} km/h, "
                f"and delivers {car.get('mileage_km_l', 'N/A')} km/l mileage. "
                f"Safety is rated {car.get('ncap', 'N/A')}/5 (NCAP). "
                f"{extract_description(car['text'])}"
            )
        state["answer"] = "\n\n".join(lines)
    except Exception as e:
        state["answer"] = f"❌ Error in car_agent: {str(e)}"
    return state
# === Test block ===
# if __name__ == "__main__":
#     state = {"user_query": "What's the most efficient and safe car?"}
#     result = car_agent(state)
#     print(result["answer"])
