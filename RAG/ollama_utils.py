import requests
import subprocess
import time
import re

def check_ollama_running():
    """Check if Ollama service is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def wait_for_ollama(max_wait=30):
    """Wait for Ollama service to start"""
    print("Waiting for Ollama service to start...")
    for i in range(max_wait):
        if check_ollama_running():
            print("Ollama service is running!")
            return True
        time.sleep(1)
        if i % 5 == 0:
            print(f"Still waiting... ({i+1}/{max_wait})")
    return False

def setup_ollama():
    """Setup Ollama and pull Qwen2.5:7B model"""
    print("Setting up Ollama...")
    
    # Check if Ollama is already installed
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        print(f"Ollama already installed: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Installing Ollama...")
        try:
            subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
            print("Ollama installed successfully")
        except subprocess.CalledProcessError:
            print("Failed to install Ollama")
            return False
    
    # Check if service is already running
    if check_ollama_running():
        print("Ollama service is already running!")
    else:
        # Start Ollama service in background
        try:
            print("Starting Ollama service...")
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for service to start
            if not wait_for_ollama():
                print("Failed to start Ollama service within timeout")
                return False
                
        except Exception as e:
            print(f"Failed to start Ollama service: {e}")
            return False
    
    # Check if model is already available
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if "qwen2.5:7b" in result.stdout:
            print("Qwen2.5:7B model already available!")
            return True
    except:
        pass
    
    # Pull the model
    try:
        print("Pulling Qwen2.5:7B model (this may take several minutes)...")
        subprocess.run(["ollama", "pull", "qwen2.5:7b"], check=True)
        print("Qwen2.5:7B model downloaded successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to pull Qwen2.5:7B model")
        # Try smaller model as fallback
        try:
            print("Trying smaller model qwen2.5:1.5b as fallback...")
            subprocess.run(["ollama", "pull", "qwen2.5:1.5b"], check=True)
            print("Fallback model downloaded successfully!")
            return True
        except:
            print("Failed to pull any Qwen model")
            return False

def generate_with_ollama(prompt: str, model: str = "qwen2.5:7b") -> str:
    """Generate text using Ollama"""
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Lower temperature for more consistent answers
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(url, json=data, timeout=60)  # Increased timeout
        if response.status_code == 200:
            return response.json()["response"]
        else:
            # Try fallback model if main model fails
            if model == "qwen2.5:7b":
                print("Trying fallback model...")
                data["model"] = "qwen2.5:1.5b"
                response = requests.post(url, json=data, timeout=60)
                if response.status_code == 200:
                    return response.json()["response"]
            return f"Error: HTTP {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}"
