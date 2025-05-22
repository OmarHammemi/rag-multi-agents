import re
import sympy

# === Math agent ===
def math_agent(state: dict) -> dict:
    try:
        query = state["user_query"]

        # Extract a clean expression
        expression = extract_math_expression(query)

        if not expression:
            state["answer"] = "❌ Sorry, I couldn't identify a valid math problem in your query."
            return state

        # Parse and evaluate safely
        result = sympy.sympify(expression)

        # If it's a number, format it nicely
        if isinstance(result, (sympy.Float, float)):
            result_str = f"{float(result):.3f}".rstrip('0').rstrip('.')
        elif isinstance(result, (sympy.Integer, int)):
            result_str = str(result)
        else:
            # Fallback: call evalf() if needed
            evaluated = result.evalf()
            result_str = f"{float(evaluated):.3f}".rstrip('0').rstrip('.')
        state["answer"] = f"🧮 The result of `{expression}` is: **{result_str}**"
    except Exception as e:
        state["answer"] = f"❌ Error in math_agent: {str(e)}"

    return state

# === Extract math expression from natural language ===
def extract_math_expression(query: str) -> str:
    q = query.lower()

    # Remove filler words like "what is", "calculate", etc.
    q = re.sub(r"\b(what|is|calculate|please|solve|find|result of|equals|answer)\b", "", q)

    # Natural language replacements
    q = q.replace("plus", "+").replace("minus", "-")
    q = q.replace("times", "*").replace("multiplied by", "*")
    q = q.replace("divided by", "/").replace("over", "/")
    q = q.replace("to the power of", "**").replace("squared", "**2").replace("cubed", "**3")

    # Roots
    q = re.sub(r"square root of (\d+)", r"sqrt(\1)", q)
    q = re.sub(r"cube root of (\d+)", r"(\1)**(1/3)", q)

    # Allow safe characters: numbers, math operators, sqrt, etc.
    cleaned = re.sub(r"[^0-9a-z\+\-\*/\^\(\)\.\s]", "", q)

    return cleaned.strip()

# # # === Test block ===
# if __name__ == "__main__":
#     test_queries = [
#         "What is (5 + 3) * 2 squared minus 4?",
#         "4 to the power of 3",
#         "square root of 81",
#         "cube root of 27 plus 3",
#         "12 plus 5 divided by 3",
#         "What is 10 minus 4 squared?"
#     ]

#     for q in test_queries:
#         state = {"user_query": q}
#         result = math_agent(state)
#         print(f"{q} → {result['answer']}")
