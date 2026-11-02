""" Composition root that constructs and shares application services. """

from config import settings
from services.ollama_service import OllamaService
from services.audit_service import AuditService
from services.document_service import DocumentService
from services.retrieval_service import RetrievalService, RAGService
from services.recovery_service import RecoveryService
from repositories.qdrant_repository import QdrantRepository

_services = None

def get_services():
    """ Build shared application services and run startup recovery once per process. """
    global _services
    if _services is None:
        # Fail fast on invalid environment settings before opening clients.
        settings.validate()
        # Initialize durable metadata before repositories that depend on it.
        audit = AuditService(settings.sqlite_path)
        ollama = OllamaService(
            settings.ollama_url,
            settings.connect_timeout,
            settings.read_timeout
        )
        qdrant = QdrantRepository(settings.qdrant_url, audit)

        documents = DocumentService(
            ollama,
            qdrant,
            audit,
            settings.embedding_batch_size,
            settings.max_upload_mb * 1024 * 1024,
            settings.max_text_mb * 1024 * 1024
        )
        retrieval = RetrievalService(ollama, qdrant, audit)
        rag = RAGService(ollama, retrieval, audit)
        recovery = RecoveryService(audit, qdrant)

        # Best-effort startup recovery: application can still start if Qdrant
        # is temporarily unavailable; unresolved manifests remain retryable.
        try:
            recovery_report = recovery.run(stale_after_seconds=settings.recovery_stale_seconds)
        except Exception as exc:
            audit.record_event(
                "STARTUP_RECOVERY", "FAILED",
                {"error": str(exc)}
            )
            recovery_report = {
                "claimed": 0, "committed": 0, "rolled_back": 0,
                "failed": 1, "errors": [str(exc)]
            }

        _services = {
            "ollama": ollama,
            "qdrant": qdrant,
            "audit": audit,
            "documents": documents,
            "retrieval": retrieval,
            "rag": rag,
            "recovery": recovery,
            "recovery_report": recovery_report
        }
    return _services
