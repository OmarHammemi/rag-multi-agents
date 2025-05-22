import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import TypedDict
from langgraph.graph import StateGraph

# Import agents and router
from agents.car_agent import car_agent
from agents.country_agent import country_agent
from agents.math_agent import math_agent
from utils.router import route_query

# === Define shared state ===
class RAGState(TypedDict):
    user_query: str
    answer: str

# === Router Node ===
def router_node(state: RAGState) -> dict:
    task = route_query(state["user_query"])
    print(f"[Router] Task: {task}")
    return {
        # Return the next node in the 'next' key
        "next": {
            "car": "car_agent",
            "country": "country_agent",
            "math": "math_agent"
        }.get(task, "fallback")
    }

# === Agent Nodes ===
# These should be modified in their respective files to return dicts
# For example in car_agent.py:
# def car_agent(state: RAGState) -> dict:
#     answer = "Your car answer here"
#     return {"answer": answer}

# === Fallback handler ===
def fallback_node(state: RAGState) -> dict:
    return {"answer": "❌ Sorry, I couldn't understand your question."}

# === Build LangGraph ===
def build_langgraph():
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("car_agent", car_agent)
    graph.add_node("country_agent", country_agent)
    graph.add_node("math_agent", math_agent)
    graph.add_node("fallback", fallback_node)

    # Entry point
    graph.set_entry_point("router")

    # Conditional edges based on router's 'next' output
    graph.add_conditional_edges(
        "router",
        lambda state: state["next"],
        {
            "car_agent": "car_agent",
            "country_agent": "country_agent",
            "math_agent": "math_agent",
            "fallback": "fallback"
        }
    )

    # Define end points
    graph.set_finish_point("car_agent")
    graph.set_finish_point("country_agent")
    graph.set_finish_point("math_agent")
    graph.set_finish_point("fallback")

    return graph.compile()

# === Example Usage ===
if __name__ == "__main__":
    rag_bot = build_langgraph()

    examples = [
        "Tell me about a coupe with high horsepower",
        "What is the capital of Tunisia?",
        "What is (6 + 4) squared minus 3?",
        "What's the weather in Tokyo today?"
    ]

    for query in examples:
        result = rag_bot.invoke({"user_query": query})
        print(f"\n🔍 Query: {query}\n💬 Answer: {result['answer']}")