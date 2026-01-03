
import os
import warnings
# Suppress Pydantic V1 compatibility warning on Python 3.14
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
import time

# --- Configuration ---
LLM_MODEL = "llama3.2:1b" 

# Safe LLM Wrapper
class SafeChatOllama:
    def __init__(self, model, temperature=0.7):
        self.llm = ChatOllama(model=model, temperature=temperature)
        self.model = model
        self.is_mock = False

    def invoke(self, messages):
        try:
            return self.llm.invoke(messages)
        except Exception as e:
            # Fallback Mock Response
            print(f"Connection Error: {e}. Switching to Mock Mode.")
            self.is_mock = True
            
            error_msg = str(e)
            suggestion = "Please check if Ollama is running."
            if "404" in error_msg:
                suggestion = f"Model '{self.model}' not found. Run `ollama pull {self.model}` in your terminal."
            elif "refused" in error_msg:
                suggestion = "Ollama connection refused. Run `ollama serve`."
            
            content = f"⚠️ **Mock Response** (System Error): {error_msg}\n\n**Fix**: {suggestion}\n\nThis is a simulated response."
            
            # Simple keyword matching to make mock less generic
            msg_str = str(messages[0].content).lower()
            if "necessity" in msg_str:
                content = f"⚠️ **Mock Analysis**: (Error: {suggestion})\n\nBased on standard necessity criteria, this item appears to be a **WANT**."
            elif "budget" in msg_str:
                content = f"⚠️ **Mock Budget**: (Error: {suggestion})\n\nYour budget seems reasonable for a mid-range option."
            elif "quality" in msg_str:
                content = f"⚠️ **Mock Quality**: (Error: {suggestion})\n\nKey quality markers include build materials and brand reputation."
            elif "alternative" in msg_str:
                content = f"⚠️ **Mock Alternatives**: (Error: {suggestion})\n\n1. **Option A**: A budget competitor.\n2. **Option B**: A premium alternative."
            elif "recommendation" in msg_str:
                content = f"⚠️ **Mock Verdict**: (Error: {suggestion})\n\n**WAIT**. Check your local LLM configuration."
            
            # wrapper to match object interface
            class MockResponse:
                def __init__(self, content):
                    self.content = content
            return MockResponse(content)

llm = SafeChatOllama(model=LLM_MODEL, temperature=0.7)

# --- State Definition ---
class ShoppingState(TypedDict):
    product: str
    budget: str
    purpose: str
    preferences: str
    
    # Outputs from nodes
    necessity_analysis: str
    budget_evaluation: str
    quality_assessment: str
    alternatives: str
    final_recommendation: str

# --- Nodes ---

def analyze_necessity(state: ShoppingState):
    """Determines if the purchase is a need or a want."""
    prompt = f"""
    You are a wise shopping assistant.
    User wants to buy: {state['product']}
    Purpose: {state['purpose']}
    
    Analyze if this purchase is a NECESSITY or a WANT based on the purpose. 
    Explain why in 2-3 sentences.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"necessity_analysis": response.content}

def evaluate_budget(state: ShoppingState):
    """Checks if the budget is realistic."""
    prompt = f"""
    User wants to buy: {state['product']}
    Budget: {state['budget']}
    
    Is this budget realistic for a good quality version of this product?
    If it's too low, warn them. If it's generous, mention that.
    Provide 1 cost-saving tip.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"budget_evaluation": response.content}

def assess_quality(state: ShoppingState):
    """Evaluates what makes a good quality version of this product."""
    prompt = f"""
    User Product: {state['product']}
    Preferences: {state['preferences']}
    
    What are the key markers of QUALITY for this product? 
    What determines durability and value for money?
    Keep it brief (bullet points).
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"quality_assessment": response.content}

def suggest_alternatives(state: ShoppingState):
    """Suggests alternatives."""
    prompt = f"""
    User wants: {state['product']} for {state['purpose']}.
    Budget: {state['budget']}
    
    Suggest 2 alternatives:
    1. A Cheaper option that still does the job.
    2. A Better/Different option they might not have thought of.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"alternatives": response.content}

def finalize_recommendation(state: ShoppingState):
    """Synthesizes everything into a final verdict."""
    prompt = f"""
    Synthesize the following into a final recommendation for the user.
    
    Product: {state['product']}
    Necessity: {state['necessity_analysis']}
    Budget Check: {state['budget_evaluation']}
    Quality Check: {state['quality_assessment']}
    Alternatives: {state['alternatives']}
    
    Give a Final Verdict: Buy, Wait, or Buy Alternative?
    Keep it friendly and decisive.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final_recommendation": response.content}

# --- Graph Construction ---
workflow = StateGraph(ShoppingState)

workflow.add_node("necessity", analyze_necessity)
workflow.add_node("budget", evaluate_budget)
workflow.add_node("quality", assess_quality)
workflow.add_node("alternatives", suggest_alternatives)
workflow.add_node("finalize", finalize_recommendation)

# Linear flow
workflow.set_entry_point("necessity")
workflow.add_edge("necessity", "budget")
workflow.add_edge("budget", "quality")
workflow.add_edge("quality", "alternatives")
workflow.add_edge("alternatives", "finalize")
workflow.add_edge("finalize", END)

app_graph = workflow.compile()

if __name__ == "__main__":
    print("Running test invocation...")
    initial_state = {
        "product": "Test Product",
        "budget": "$100",
        "purpose": "Testing",
        "preferences": "None",
        "necessity_analysis": "",
        "budget_evaluation": "",
        "quality_assessment": "",
        "alternatives": "",
        "final_recommendation": ""
    }
    try:
        result = app_graph.invoke(initial_state)
        print("Test Result Keys:", result.keys())
        print("Test Successful!")
    except Exception as e:
        print(f"Test Failed: {e}")

