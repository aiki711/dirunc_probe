import os
import json
import time
from pathlib import Path
from google import genai
from google.api_core import exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

def load_api_key():
    env_path = Path(".env")
    if not env_path.exists():
        return os.environ.get("GOOGLE_API_KEY")
    with env_path.open("r") as f:
        for line in f:
            if line.startswith("GOOGLE_API_KEY="):
                return line.strip().split("=", 1)[1].strip("'\"")
    return None

# Retry logic for 429/503 errors
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=10, max=60),
    retry=(retry_if_exception_type(exceptions.ResourceExhausted) | retry_if_exception_type(exceptions.ServiceUnavailable))
)
def generate_with_retry(client, model_id, prompt):
    return client.models.generate_content(model=model_id, contents=prompt)

def main():
    api_key = load_api_key()
    if not api_key:
        print("API Key not found in .env or environment variables.")
        return

    client = genai.Client(api_key=api_key)
    # Using the most standard stable flash model
    model_id = "gemini-flash-latest" 

    context = "User: I need a train departing from Cambridge to Stansted Airport.\nAssistant: I can help you with that. What time would you like to depart?"
    filled_text = "I would like to depart Cambridge after 11:00 please."
    role = "Source"
    dropped_span = "Cambridge"

    prompt_template = """You are a linguist assisting in Case Grammar research.
Task: Create a "Missing" version of the "Filled Sentence" where the specific information "{dropped_span}" (Role: {role}) is omitted or replaced by a context-appropriate placeholder.

Context: 
{context}

Filled Sentence: {filled_text}
Omit information: {dropped_span}

Instructions:
1. The "Missing" version MUST be perfectly natural and fluent English, sounding like a typical follow-up or reply.
2. The specific value "{dropped_span}" MUST be removed.
3. Instead of the value, use a context-appropriate placeholder (like "there", "that place", "the city", "it", etc.) if it sounds more natural than complete omission.
4. The goal is to make the information "critically missing" or "vague" so that the model's internal uncertainty is maximized.

Create two versions:
Version A (Soft Placeholder): Replace with something like "that city", "that time", etc.
Version B (Strong Placeholder/Omission): Replace with "there", "it", or complete omission if it sounds natural.

Output in JSON format:
{{
  "version_a": "...",
  "version_b": "..."
}}"""

    prompt = prompt_template.format(
        context=context,
        filled_text=filled_text,
        role=role,
        dropped_span=dropped_span
    )

    print(f"Requesting to {model_id} with retry logic...")
    try:
        response = generate_with_retry(client, model_id, prompt)
        text = response.text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        result = json.loads(text)
        
        print("\n--- Pilot Test Result ---")
        print(f"Version A (Soft): {result['version_a']}")
        print(f"Version B (Strong): {result['version_b']}")
        
    except exceptions.ResourceExhausted:
        print("Error: API Quota exhausted (429) even after retries. Please check your Google AI Studio quota.")
    except Exception as e:
        print("An unexpected error occurred:", e)

if __name__ == "__main__":
    main()
