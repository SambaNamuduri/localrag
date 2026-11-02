"""Central configuration for endpoints, timeouts, storage, and resource limits."""
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment variables."""
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/")
    sqlite_path: str = os.getenv("AUDIT_DB_PATH", "data/ai_activity.sqlite3")
    connect_timeout: int = int(os.getenv("CONNECT_TIMEOUT", "5"))
    read_timeout: int = int(os.getenv("READ_TIMEOUT", "180"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "30"))
    max_text_mb: int = int(os.getenv("MAX_TEXT_MB", "50"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    recovery_stale_seconds: int = int(os.getenv("RECOVERY_STALE_SECONDS", "120"))
    default_collection: str = os.getenv("DEFAULT_COLLECTION", "technical_docs")

    def validate(self):
        """Reject invalid resource limits and missing service endpoints."""
        values = {
            "CONNECT_TIMEOUT": self.connect_timeout,
            "READ_TIMEOUT": self.read_timeout,
            "MAX_UPLOAD_MB": self.max_upload_mb,
            "MAX_TEXT_MB": self.max_text_mb,
            "EMBEDDING_BATCH_SIZE": self.embedding_batch_size,
        }
        invalid = [key for key, value in values.items() if value <= 0]
        if invalid:
            raise ValueError("Values must be positive: " + ", ".join(invalid))
        if self.recovery_stale_seconds < 0:
            raise ValueError("RECOVERY_STALE_SECONDS cannot be negative.")
        if not self.ollama_url or not self.qdrant_url:
            raise ValueError("OLLAMA_URL and QDRANT_URL must not be empty.")

settings = Settings()
