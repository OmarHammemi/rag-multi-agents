import re
from typing import Literal

# Define supported task types
TaskType = Literal["car", "country", "math", "unknown"]

def route_query(query: str) -> TaskType:
    q = query.lower()

    # === CAR keywords ===
    car_keywords = [
        "engine", "mileage", "horsepower", "top speed", "fuel efficiency", "suv", "sedan", "coupe",
        "convertible", "hatchback", "vehicle", "car", "ncap", "model", "brand", "km/l", "kmph", "car specs",
        "acceleration", "electric car", "gasoline", "transmission", "launch year"
    ]
    if any(word in q for word in car_keywords):
        return "car"

    # === COUNTRY keywords ===
    country_keywords = [
        "capital", "population", "language", "area", "square kilometers", "country", "national animal",
        "national bird", "flag", "rivers", "continent", "geography", "official language", "currency",
        "borders", "neighbors", "government", "president", "prime minister"
    ]
    if any(word in q for word in country_keywords):
        return "country"

    # === MATH patterns & keywords ===
    math_patterns = [
        r"\b(plus|minus|times|multiplied by|divided by|over|squared|cubed|power|square root|cube root)\b",
        r"\d+\s*[-+*/^]\s*\d+",        # basic math operators
        r"\d+\s*\(.*\)",              # function-like use
        r"\d+\s*\^\s*\d+"            # exponentiation
    ]
    if any(re.search(pat, q) for pat in math_patterns):
        return "math"

    return "unknown"

# # === Optional: Test locally ===
# if __name__ == "__main__":
#     examples = [
#         "Tell me about the horsepower of a coupe",
#         "What's the capital of Germany?",
#         "Calculate (4 + 3) squared",
#         "How much is 5 times 8 minus 3?",
#         "Who is the president of Canada?",
#         "Show me electric vehicles with top speed",
#         "What is 12 plus 7 divided by 3?",
#         "What is the official language of Brazil?",
#         "What's the weather like today?"
#     ]
#     for q in examples:
#         print(f"{q} → {route_query(q)}")
