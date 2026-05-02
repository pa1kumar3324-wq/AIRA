"""
core/llm.py

Ollama Mistral LLM integration for AIRA.
Sends conversation history + sentiment-aware system prompt to the local Ollama instance
and streams the response back token by token.

Requirements:
    - Ollama installed and running  →  https://ollama.com
    - Mistral model pulled          →  ollama pull mistral

Ollama runs on http://localhost:11434 by default.
"""

import json
import requests
from typing import Generator
from pathlib import Path

CONFIG_FILE = Path(__file__).parent.parent / "config.json"

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {
        "ollama_url": "http://localhost:11434/api/chat",
        "model": "mistral",
        "temperature": 0.75,
        "max_tokens": 1024,
    }

config = load_config()

OLLAMA_URL   = config["ollama_url"]
MODEL_NAME   = config["model"]
TEMPERATURE  = config["temperature"]
MAX_TOKENS   = config["max_tokens"]

AIRA_BASE_SYSTEM = """You are AIRA (Adaptive Intelligent Responsive Agent), an emotionally intelligent AI chatbot.

Your core behaviour:
- You adapt your tone and language based on the user's emotional state, which is provided to you.
- You are warm, thoughtful, and conversational — never robotic or overly formal.
- You keep responses concise unless the user clearly wants depth.
- You never dismiss or invalidate feelings. You acknowledge them first, then help.
- You do not pretend to be human, but you do genuinely care about the person you're talking to.

{tone_instruction}
"""


class LLMClient:
    def __init__(self, model: str = MODEL_NAME):
        self.model = model
        self.base_url = OLLAMA_URL

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(self.base_url.replace("/api/chat", "/api/tags"), timeout=5)
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> list[str]:
        """List available models."""
        try:
            response = requests.get(self.base_url.replace("/api/chat", "/api/tags"), timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except:
            return []

    def _build_system_prompt(self, tone_hint: str) -> str:
        tone_instruction = f"Current emotional context: {tone_hint}"
        return AIRA_BASE_SYSTEM.format(tone_instruction=tone_instruction)

    def chat(
        self,
        messages: list[dict],
        tone_hint: str = "Be helpful and conversational.",
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """
        Send a conversation to Ollama and yield response tokens as they stream in.

        Args:
            messages    List of {"role": "user"|"assistant", "content": str}
            tone_hint   Sentiment-derived instruction injected into the system prompt
            stream      Whether to stream the response (default True)

        Yields:
            str — individual token chunks from the model
        """
        system_prompt = self._build_system_prompt(tone_hint)

        payload = {
            "model":    self.model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "stream":   stream,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS,
            },
        }

        try:
            response = requests.post(self.base_url, json=payload, stream=stream, timeout=60)
            response.raise_for_status()

            if stream:
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
            else:
                data = response.json()
                yield data["message"]["content"]

        except requests.exceptions.ConnectionError:
            yield (
                "\n[AIRA] Could not connect to Ollama. "
                "Make sure Ollama is running: `ollama serve`\n"
            )
        except requests.exceptions.Timeout:
            yield "\n[AIRA] The model took too long to respond. Please try again.\n"
        except Exception as e:
            yield f"\n[AIRA] Unexpected error: {e}\n"

    def is_available(self) -> bool:
        """Check whether the Ollama server is reachable."""
        try:
            resp = requests.get("http://localhost:11434", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return a list of locally available Ollama models."""
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
