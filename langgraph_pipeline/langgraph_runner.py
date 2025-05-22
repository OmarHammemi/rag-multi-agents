# agent_system.py
import os
import sys
from typing import Dict, Any
from dotenv import load_dotenv

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Load environment variables
load_dotenv(os.path.join(project_root, '.env'))

# Import your existing components
from agents.car_agent import car_agent
from agents.country_agent import country_agent
from agents.math_agent import math_agent
from utils.router import route_query

class AgentSystem:
    def __init__(self):
        """Initialize the agent system with all available agents"""
        self.agents = {
            'car': car_agent,
            'country': country_agent,
            'math': math_agent
        }
    
    def process_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the user query using the router to select the appropriate agent
        """
        query = state.get("user_query", "").strip()
        if not query:
            state["answer"] = "❌ Please provide a valid query."
            return state
        
        # Use your existing router to determine the agent
        agent_type = route_query(query)
        
        if agent_type == "unknown":
            # Fallback checks for country names
            from agents.country_agent import country_metadata
            for entry in country_metadata:
                if entry["id"].lower() in query.lower():
                    agent_type = "country"
                    break
            
            if agent_type == "unknown":
                state["answer"] = "❌ I couldn't determine what you're asking about. Please try being more specific."
                return state
        
        try:
            # Get the appropriate agent function
            agent_func = self.agents.get(agent_type)
            if not agent_func:
                state["answer"] = f"❌ No agent available to handle {agent_type} queries."
                return state
            
            # Process with the selected agent (maintaining all your existing agent logic)
            return agent_func(state)
        
        except Exception as e:
            state["answer"] = f"❌ Error processing your query: {str(e)}"
            return state

# === Example usage ===
if __name__ == "__main__":
    agent_system = AgentSystem()
    
    test_queries = [
        "What's the fastest car with good mileage?",
        "What is the capital of France?",
        "Calculate 25 plus 37 all divided by 2",
        "Tell me about Toyota Washington",
        "Population of Japan",
        "What is 5 to the power of 3?",
        "Tell me about Tunisia",  # Will trigger country fallback
        "What's the weather today?"  # Will return unknown
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        state = {"user_query": query}
        result = agent_system.process_query(state)
        print("Answer:", result["answer"])