import re
import sympy
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def math_agent(state: dict) -> dict:
    try:
        query = state["user_query"]
        
        # First check if this is actually a math problem
        if not is_math_problem(query):
            state["answer"] = "❌ This doesn't appear to be a math problem. Please ask a question with numbers and mathematical operations."
            return state
            
        # Convert natural language to math expression
        expression = convert_to_math_expression(query)
        if not expression:
            state["answer"] = "❌ Couldn't convert this to a solvable math expression."
            return state

        # Parse and evaluate safely
        result = sympy.sympify(expression)

        # Format the result
        if isinstance(result, (sympy.Float, float)):
            result_str = f"{float(result):.3f}".rstrip('0').rstrip('.')
        elif isinstance(result, (sympy.Integer, int)):
            result_str = str(result)
        else:
            evaluated = result.evalf()
            result_str = f"{float(evaluated):.3f}".rstrip('0').rstrip('.')
            
        state["answer"] = f"🧮 The result of `{expression}` is: **{result_str}**"
    except Exception as e:
        state["answer"] = f"❌ Error in math_agent: {str(e)}"
    return state

def convert_to_math_expression(query: str) -> str:
    """Convert natural language to math expression using OpenAI with fallback"""
    try:
        # First try with OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Convert this to a Python-compatible math expression only. Return just the expression."},
                {"role": "user", "content": query}
            ],
            temperature=0.1,
            max_tokens=50
        )
        expression = response.choices[0].message.content.strip()
        
        # Basic validation
        if any(c.isalpha() and c not in ['s', 'q', 'r', 't'] for c in expression):  # Allow sqrt
            raise ValueError("Invalid characters in expression")
            
        return expression
    except:
        # Fallback to traditional conversion
        return traditional_math_conversion(query)

def traditional_math_conversion(query: str) -> str:
    """Fallback conversion method"""
    q = query.lower()
    q = re.sub(r"\b(what|is|calculate|please|solve|find|result of|equals|answer)\b", "", q)
    q = q.replace("plus", "+").replace("minus", "-")
    q = q.replace("times", "*").replace("multiplied by", "*")
    q = q.replace("divided by", "/").replace("over", "/")
    q = q.replace("to the power of", "**").replace("squared", "**2").replace("cubed", "**3")
    q = re.sub(r"square root of (\d+)", r"sqrt(\1)", q)
    q = re.sub(r"cube root of (\d+)", r"(\1)**(1/3)", q)
    q = re.sub(r"[^0-9a-z\+\-\*/\^\(\)\.\s]", "", q)
    return q.strip()

def is_math_problem(query: str) -> bool:
    """Determine if the query contains math operations"""
    math_keywords = [
        'plus', 'minus', 'times', 'divided by', 'over', 
        'squared', 'cubed', 'power of', 'sqrt', 'root',
        'percent', 'of', 'sum', 'product', 'difference',
        'add', 'subtract', 'multiply', 'divide', 'modulo',
        'calculate', 'compute', 'solve', 'equals', '=',
        '+', '-', '*', '/', '^'
    ]
    return (any(c.isdigit() for c in query) and 
            any(term in query.lower() for term in math_keywords))
