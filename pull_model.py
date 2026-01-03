
import requests
import json
import sys
import os

# Set stdout to handle utf-8 even in Windows console
sys.stdout.reconfigure(encoding='utf-8')

def pull_model(model_name="llama3.2:1b"):
    url = "http://localhost:11434/api/pull"
    payload = {"name": model_name, "stream": True}
    
    print(f"Starting download for model: {model_name}")
    print("This is a smaller model (~1.3GB) optimized for your system RAM.")
    
    try:
        with requests.post(url, json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"Error: {response.text}")
                return

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    
                    # Print progress if available
                    if "total" in data and "completed" in data:
                        percent = (data["completed"] / data["total"]) * 100
                        print(f"\r{status}: {percent:.2f}%", end="")
                    else:
                        print(f"\r{status}", end="")
                        
        print("\n\nDownload Complete! You can now use the Smart Shopping Advisor.")
        
    except Exception as e:
        print(f"\nFailed to connect to Ollama: {e}")
        print("Please ensure Ollama is running.")

if __name__ == "__main__":
    pull_model()
