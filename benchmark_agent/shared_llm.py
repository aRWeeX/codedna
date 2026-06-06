import os
import time
from pathlib import Path

# The Unified Model Registry
MODELS = {
    # Standard Models
    "gemini-2.5-flash":       {"provider": "gemini", "model_id": "gemini-2.5-flash"},
    "gemini-2.5-pro":         {"provider": "gemini", "model_id": "gemini-2.5-pro"},
    
    # 3.x Reasoning Models
    "gemini-3.1-flash-lite":  {"provider": "gemini", "model_id": "gemini-3.1-flash-lite"},
    "gemini-3.5-flash":       {"provider": "gemini", "model_id": "gemini-3.5-flash"},
    "gemini-3.1-pro-preview": {"provider": "gemini", "model_id": "gemini-3.1-pro-preview"},
}

FORCE_FINAL_PROMPT = "Please provide your final answer."

def call_with_retry(client, model_id, contents, config, retries=3):
    for attempt in range(retries):
        try:
            return client.models.generate_content(model=model_id, contents=contents, config=config)
        except Exception as e:
            print(f"API Error: {e}. Retrying...")
            if attempt == retries - 1:
                raise e
            time.sleep(2 ** attempt)

# NEW: Safely extracts text AND reasoning payloads
def extract_text_safely(parts):
    if not parts:
        return ""
    extracted = []
    for p in parts:
        if getattr(p, "text", None):
            extracted.append(p.text)
        elif getattr(p, "thought", None):
            extracted.append(p.thought)
    return "".join(extracted)

def run_gemini(problem_statement: str, repo_path: Path, model_name: str):
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return ["Error: google-genai not installed"], ""

    model_id = MODELS[model_name]["model_id"]
    project = os.environ.get("VERTEX_PROJECT")
    location = os.environ.get("VERTEX_LOCATION", "global")
    
    if not project:
        return ["Error: VERTEX_PROJECT environment variable missing"], ""
        
    client = genai.Client(vertexai=True, project=project, location=location)
    
    config = types.GenerateContentConfig(
        max_output_tokens=8000,
        system_instruction="You are an expert software engineer resolving a GitHub issue. Do not only return reasoning; you must output text."
    )
    
    prompt = f"Problem: {problem_statement}"
    response = call_with_retry(client, model_id, prompt, config)
    
    log = [f"User: {prompt}"]
    
    # Extract parts safely to avoid crashes
    parts = []
    if hasattr(response, 'candidates') and response.candidates:
        if hasattr(response.candidates[0], 'content') and response.candidates[0].content:
            parts = response.candidates[0].content.parts
            
    final_text = extract_text_safely(parts)
    log.append(f"Model: {final_text}")
    
    return log, (final_text or "")
