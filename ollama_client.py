from __future__ import annotations

from typing import Any

import httpx

from app.config import Settings


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.ollama_base_url
        self.timeout = httpx.Timeout(300.0, connect=5.0)
        self.http = httpx.Client(timeout=self.timeout)

    def models(self) -> set[str]:
        try:
            response = self.http.get(f"{self.base_url}/api/tags", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.base_url}. Start the Ollama app and retry."
            ) from exc
        return {
            str(item.get("name", ""))
            for item in response.json().get("models", [])
            if item.get("name")
        }

    def missing_models(self) -> list[str]:
        available = self.models()
        required = {self.settings.ollama_chat_model, self.settings.ollama_embedding_model}
        normalized = {name.removesuffix(":latest") for name in available}
        return sorted(model for model in required if model not in available and model.removesuffix(":latest") not in normalized)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), 32):
            try:
                response = self.http.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": self.settings.ollama_embedding_model,
                        "input": texts[start : start + 32],
                        "truncate": True,
                        "keep_alive": self.settings.ollama_keep_alive,
                    },
                    headers={"Connection": "keep-alive"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise OllamaError(
                    f"Ollama embeddings failed for model '{self.settings.ollama_embedding_model}'. "
                    "Check that the model is installed and Ollama is running."
                ) from exc
            batch = response.json().get("embeddings", [])
            if len(batch) != len(texts[start : start + 32]):
                raise OllamaError("Ollama returned an unexpected number of embeddings.")
            embeddings.extend(batch)
        return embeddings

    def embed(self, text: str) -> list[float]:
        result = self.embed_many([text])
        if not result or not result[0]:
            raise OllamaError("Ollama returned an empty embedding.")
        return result[0]

    def chat_json(self, messages: list[dict[str, str]]) -> str:
        try:
            response = self.http.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.settings.ollama_chat_model,
                    "messages": messages,
                    "format": "json",
                    "stream": False,
                    "keep_alive": self.settings.ollama_keep_alive,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": self.settings.ollama_num_predict,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Ollama generation failed for model '{self.settings.ollama_chat_model}'. "
                "Check that the model is installed and Ollama is running."
            ) from exc
        message: dict[str, Any] = response.json().get("message", {})
        content = message.get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise OllamaError("Ollama returned an empty response.")
        return content