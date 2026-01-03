
import sys
# Set stdout to handle utf-8 even in Windows console
sys.stdout.reconfigure(encoding='utf-8')

import time
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def test_connection():
    print("Test 1: Testing Raw HTTP Connectivity...")
    import requests
    try:
        res = requests.get('http://localhost:11434/api/tags')
        print(f"HTTP Status: {res.status_code}")
        print(f"Available Models: {res.text}")
    except Exception as e:
        print(f"HTTP Failed: {e}")

    print("\nTest 2: Testing LangChain ChatOllama...")
    try:
        llm = ChatOllama(model="llama3", temperature=0.7)
        print("Invoking ChatOllama...")
        res = llm.invoke([HumanMessage(content="Hello, are you there?")])
        print(f"Success! Response: {res.content}")
    except Exception as e:
        print(f"ChatOllama Failed: {e}")
        print(f"Error Type: {type(e)}")

if __name__ == "__main__":
    test_connection()
