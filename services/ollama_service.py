""" Ollama HTTP client and dynamic model registry integration. """

from dataclasses import dataclass, asdict, field
from typing import Any
import logging
import requests

logger = logging.getLogger(__name__)

class OllamaError(Exception):
    """Base exception for Ollama integration failures."""


class OllamaConnectionError(OllamaError):
    """Raised when Ollama cannot be reached or times out."""


class OllamaModelNotFoundError(OllamaError):
    """Raised when the requested Ollama model does not exist."""


class OllamaRequestError(OllamaError):
    """Raised when Ollama rejects a request or returns an HTTP error."""


class OllamaInvalidResponseError(OllamaError):
    """Raised when Ollama returns a malformed or incomplete response."""


class OllamaCapabilityError(OllamaError):
    """Reserved for model capability validation failures."""

@dataclass
class OllamaModel:
    """ OllamaModel groups related application behavior. """
    name: str
    size: int = 0
    digest: str = ""
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""
    capabilities: list[str] = field(default_factory=list)
    installed: bool = True
    running: bool = False
    capability_status: str = "unknown"

    @property
    def supports_chat(self):
        """ Check whether Ollama advertises text-generation capability. """
        return "completion" in self.capabilities

    @property
    def supports_embeddings(self):
        """ Check whether Ollama advertises embedding capability. """
        return "embedding" in self.capabilities

    def to_dict(self):
        """ Serialize model metadata and derived capability flags for the UI. """
        data = asdict(self)
        data["supports_chat"] = self.supports_chat
        data["supports_embeddings"] = self.supports_embeddings
        return data

class OllamaService:
    """ OllamaService groups related application behavior. """
    def __init__(self, base_url, connect_timeout=5, read_timeout=180):
        """ Handle the   init   operation. """
        self.base_url = base_url.rstrip("/")
        self.timeout = (connect_timeout, read_timeout)
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _request(self, method, endpoint, **kwargs):
        """ Call Ollama with timeouts and normalize transport/API failures. """
        try:
            response = self.session.request(
                method, f"{self.base_url}{endpoint}",
                timeout=self.timeout, **kwargs
            )
            if response.status_code == 404:
                raise OllamaModelNotFoundError(response.text[:500])
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError as exc:
                raise OllamaInvalidResponseError("Ollama returned invalid JSON.") from exc
            if not isinstance(data, dict):
                raise OllamaInvalidResponseError("Unexpected Ollama response structure.")
            if data.get("error"):
                raise OllamaRequestError(str(data["error"]))
            return data
        except (OllamaError,):
            raise
        except requests.ConnectionError as exc:
            raise OllamaConnectionError(f"Cannot connect to Ollama at {self.base_url}") from exc
        except requests.Timeout as exc:
            raise OllamaConnectionError("Ollama request timed out.") from exc
        except requests.HTTPError as exc:
            raise OllamaRequestError(f"Ollama HTTP error: {exc}") from exc
        except requests.RequestException as exc:
            raise OllamaRequestError(f"Ollama transport error: {exc}") from exc

    def health_check(self):
        """ Check dependency availability and return a safe status result. """
        try:
            data = self._request("GET", "/api/version")
            return {"available": True, "version": data.get("version"), "error": None}
        except OllamaError as exc:
            logger.warning("Ollama health check failed: %s", exc)
            return {"available": False, "version": None, "error": str(exc)}

    def list_installed_models(self):
        """ List all models installed in the local Ollama registry. """
        models = self._request("GET", "/api/tags").get("models")
        if not isinstance(models, list):
            raise OllamaInvalidResponseError("Invalid /api/tags response.")
        return models

    def list_running_models(self):
        """ List model names currently loaded in Ollama memory. """
        models = self._request("GET", "/api/ps").get("models", [])
        return {m.get("name") or m.get("model") for m in models if m.get("name") or m.get("model")}

    def show_model(self, name):
        """ Fetch metadata and advertised capabilities for one model. """
        if not name or not name.strip():
            raise ValueError("Model name is required.")
        return self._request("POST", "/api/show", json={"model": name, "verbose": False})

    def discover_models(self):
        """ Combine installed, running, and capability metadata into model records. """
        installed = self.list_installed_models()
        running = self.list_running_models()
        discovered = []
        for item in installed:
            name = item.get("name") or item.get("model")
            if not name:
                logger.warning("Skipping malformed model record: missing name")
                continue
            details = item.get("details") or {}
            capabilities = []
            capability_status = "unknown"
            try:
                info = self.show_model(name)
                raw = info.get("capabilities")
                if isinstance(raw, list):
                    capabilities = raw
                    capability_status = "reported"
            except OllamaError as exc:
                logger.warning("Could not inspect model %s: %s", name, exc)
            discovered.append(OllamaModel(
                name=name, size=item.get("size", 0), digest=item.get("digest", ""),
                family=details.get("family", ""), parameter_size=details.get("parameter_size", ""),
                quantization=details.get("quantization_level", ""), capabilities=capabilities,
                running=name in running, capability_status=capability_status
            ))
        return discovered

    def generate_embeddings(self, model_name, texts):
        """ Generate and validate one vector for each supplied text. """
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Embedding inputs must be a non-empty list of non-empty strings.")
        data = self._request("POST", "/api/embed", json={
            "model": model_name, "input": texts, "truncate": False
        })
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise OllamaInvalidResponseError("Embedding count does not match input count.")
        dimension = None
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise OllamaInvalidResponseError("Invalid embedding vector.")
            if not all(isinstance(v, (int, float)) for v in vector):
                raise OllamaInvalidResponseError("Embedding contains non-numeric values.")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise OllamaInvalidResponseError("Inconsistent embedding dimensions.")
        return vectors

    def chat(self, model_name, messages, temperature=0.2):
        """ Send a non-streaming chat request to the selected Ollama model. """
        if not model_name or not messages:
            raise ValueError("Model name and messages are required.")
        data = self._request("POST", "/api/chat", json={
            "model": model_name, "messages": messages, "stream": False,
            "options": {"temperature": temperature}
        })
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaInvalidResponseError("Ollama returned no assistant content.")
        return {
            "model": data.get("model", model_name),
            "content": message["content"],
            "prompt_tokens": data.get("prompt_eval_count"),
            "generated_tokens": data.get("eval_count"),
            "total_duration": data.get("total_duration")
        }

    def close(self):
        """ Release the HTTP client's resources. """
        self.session.close()
