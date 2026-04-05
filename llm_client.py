"""
Ollama LLM client for SmartPharma RAG.
Handles API communication with local Ollama instance.
"""
import requests
from config import OLLAMA_URL, OLLAMA_MODEL


def call_ollama(
    prompt: str,
    model: str = OLLAMA_MODEL,
    temperature: float = 0.2,
    max_new_tokens: int = 1024,
    num_ctx: int = 8192
) -> str:
    """
    Send prompt to Ollama API and return generated response.
    
    Args:
        prompt: Input prompt for the model
        model: Model identifier (default from config)
        temperature: Sampling temperature (0-1)
        max_new_tokens: Maximum tokens to generate
        num_ctx: Context window size
        
    Returns:
        Generated text response
        
    Raises:
        RuntimeError: If API call fails
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_new_tokens,
            "num_ctx": num_ctx
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=600)
        
        if response.status_code != 200:
            try:
                error = response.json().get("error")
            except Exception:
                error = response.text
            raise RuntimeError(f"Ollama error {response.status_code}: {error}")
        
        data = response.json()
        return data.get("response") or data.get("content") or str(data)
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to connect to Ollama: {e}")