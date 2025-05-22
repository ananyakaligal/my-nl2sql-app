# gemini_flash_beta_llm.py

import os
import requests
from dotenv import load_dotenv
from langchain.llms.base import LLM
from typing import Optional, List

load_dotenv()

# Default settings for the Flash endpoint
API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL_NAME = "gemini-2.0-flash"

class GeminiFlashBetaLLM(LLM):
    """LangChain LLM wrapper around the v1beta Flash endpoint."""
    # Pydantic fields for BaseModel compatibility
    model_name: str = DEFAULT_MODEL_NAME
    api_key: str = API_KEY

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Ensure API key
        key = self.api_key or API_KEY
        if not key:
            raise ValueError("Set your GEMINI_API_KEY in .env")

        url = f"{BASE_URL}/models/{self.model_name}:generateContent"
        params = {"key": key}
        body = {"contents": [{"parts": [{"text": prompt}]}]}

        r = requests.post(url, params=params, json=body)
        r.raise_for_status()
        data = r.json()

        # Extract the first candidate
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        first = candidates[0]

        # Handle content vs output fields
        content = first.get("content") if "content" in first else first.get("output")
        text = ""

        if isinstance(content, dict):
            # Some models return nested parts
            parts = content.get("parts") or []
            for part in parts:
                # support dict or plain text
                fragment = part.get("text") if isinstance(part, dict) else part
                if fragment:
                    text += fragment
        elif isinstance(content, str):
            text = content

        return text.strip()

    @property
    def _llm_type(self) -> str:
        return f"gemini-flash-v1beta-{self.model_name}"