import faiss
import json
import numpy as np
import re
from openai import OpenAI
from dotenv import load_dotenv
import os

# === Configuration ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MAX_RESULTS = 5  # Maximum number of results to return

# === Load Data ===
car_index = faiss.read_index("/home/omar/rag/index/car_faiss.index")
with open("/home/omar/rag/data/car_metadata.json", "r", encoding="utf-8") as f:
    car_metadata = json.load(f)

# === Helper Functions ===
def embed_query(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding).astype("float32").reshape(1, -1)

def parse_specs(text: str) -> dict:
    """Extract car specifications with improved accuracy"""
    specs = {
        "body_type": "N/A",
        "horsepower": "N/A",
        "top_speed": "N/A",
        "mileage_km_l": "N/A",
        "ncap": "N/A"
    }

    # Body Type (more robust extraction)
    if match := re.search(r"Specs:\s*([^,]+)", text, re.IGNORECASE):
        specs["body_type"] = match.group(1).strip()

    # Numeric specs with validation
    if match := re.search(r"(\d+)\s*km/h", text):
        specs["top_speed"] = int(match.group(1))
    if match := re.search(r"(\d+)\s*km/l", text):
        specs["mileage_km_l"] = int(match.group(1))
    if match := re.search(r"(\d+)\s*hp", text, re.IGNORECASE):
        specs["horsepower"] = int(match.group(1))
    if match := re.search(r"NCAP Rating:\s*([\d.]+)/5", text):
        specs["ncap"] = float(match.group(1))

    return specs

def extract_brand(text: str) -> str:
    """More reliable brand extraction"""
    patterns = [
        r"by\s+([^\(]+)\s*\(",
        r"manufactured by\s+([^,]+)",
        r"produced by\s+([^,]+)"
    ]
    for pattern in patterns:
        if match := re.search(pattern, text, re.IGNORECASE):
            return match.group(1).strip()
    return "Unknown manufacturer"

def extract_full_description(text: str) -> str:
    """Get complete description without truncation"""
    if match := re.search(r"Description:(.+?)(?:\n\n|$)", text, re.DOTALL):
        return match.group(1).strip()
    return "No description available"

def get_requested_count(query: str) -> int:
    """Precisely extract the requested number of results"""
    if match := re.search(r"(?:show|give|list|find)\s+(?:me\s+)?(\d+)\s+cars?", query.lower()):
        return min(int(match.group(1)), MAX_RESULTS)
    return 1

# === Main Agent ===
def car_agent(state: dict) -> dict:
    try:
        query = state["user_query"].lower()
        num_results = get_requested_count(query)
        
        # Semantic search
        query_embedding = embed_query(query)
        distances, indices = car_index.search(query_embedding, k=MAX_RESULTS)
        
        # Process results
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

        # Dynamic ranking
        def rank(car):
            base = -car["score"]
            if "speed" in query:
                base += 0.01 * (car["top_speed"] if isinstance(car["top_speed"], int) else 0)
            if "efficient" in query or "mileage" in query:
                base += 0.01 * (car["mileage_km_l"] if isinstance(car["mileage_km_l"], int) else 0)
            if "safe" in query:
                base += 0.1 * (car["ncap"] if isinstance(car["ncap"], float) else 0)
            if "power" in query or "hp" in query:
                base += 0.01 * (car["horsepower"] if isinstance(car["horsepower"], int) else 0)
            return base

        # Get exact number of requested results
        top_results = sorted(results, key=rank, reverse=True)[:num_results]
        
        if not top_results:
            state["answer"] = "❌ No matching cars found. Please try different search terms."
            return state

        # Build complete response paragraphs
        response = []
        for car in top_results:
            paragraph = (
                f"The {car['id']} is a {car['body_type']} manufactured by {extract_brand(car['text'])}. "
                f"It features a {car['horsepower']} HP engine with a top speed of {car['top_speed']} km/h "
                f"and fuel efficiency of {car['mileage_km_l']} km/l. "
                f"This model has a safety rating of {car['ncap']}/5 stars. "
                f"{extract_full_description(car['text'])}"
            )
            response.append(paragraph)

        state["answer"] = "\n\n".join(response) if num_results > 1 else response[0]
        
    except Exception as e:
        state["answer"] = f"❌ Error processing your request: {str(e)}"
    return state
