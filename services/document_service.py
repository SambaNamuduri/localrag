""" Document extraction, validation, chunking, and ingestion orchestration. """

import io
import re
import uuid
from pypdf import PdfReader
from docx import Document
from exceptions.app_exceptions import ValidationError

class DocumentService:
    """ DocumentService groups related application behavior. """
    ALLOWED = {".pdf", ".docx", ".txt", ".md"}

    @staticmethod
    def extract(upload, max_upload_bytes=30 * 1024 * 1024):
        """ Validate an uploaded file and extract its textual content. """
        if upload is None:
            raise ValidationError("No uploaded file supplied.")
        name = getattr(upload, "name", "upload").lower()
        suffix = "." + name.rsplit(".", 1)[-1] if "." in name else ""
        if suffix not in DocumentService.ALLOWED:
            raise ValidationError(f"Unsupported file type: {suffix}")
        # Check reported size before reading bytes, then verify actual bytes.
        reported_size = getattr(upload, "size", None)
        if reported_size is not None and reported_size > max_upload_bytes:
            raise ValidationError(
                f"File exceeds the {max_upload_bytes // (1024 * 1024)} MB upload limit."
            )
        try:
            raw = upload.getvalue()
            if len(raw) > max_upload_bytes:
                raise ValidationError(
                    f"File exceeds the {max_upload_bytes // (1024 * 1024)} MB upload limit."
                )
            if not raw:
                raise ValidationError("Uploaded file is empty.")
            if suffix == ".pdf":
                reader = PdfReader(io.BytesIO(raw), strict=True)
                if reader.is_encrypted:
                    raise ValidationError("Encrypted PDFs are not supported.")
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            elif suffix == ".docx":
                doc = Document(io.BytesIO(raw))
                text = "\n".join(p.text for p in doc.paragraphs)
            else:
                text = raw.decode("utf-8-sig", errors="strict")
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(f"Could not extract text from {name}: {exc}") from exc
        if not text.strip():
            raise ValidationError("No extractable text found. Scanned PDFs require OCR.")
        return text

    @staticmethod
    def validate_text_size(text, max_text_bytes=50 * 1024 * 1024):
        """ Reject extracted text above the configured processing limit. """
        if len(text.encode("utf-8")) > max_text_bytes:
            raise ValidationError("Extracted text exceeds the configured processing limit.")

    @staticmethod
    def chunk(text, size=2400, overlap=300):
        """ Split normalized text into overlapping chunks for indexing. """
        if not isinstance(text, str) or not text.strip():
            raise ValidationError("Document text cannot be empty.")
        if size < 100 or overlap < 0 or overlap >= size:
            raise ValidationError("Invalid chunk size or overlap.")
        normalized = re.sub(r"\s+", " ", text).strip()
        chunks, start = [], 0
        while start < len(normalized):
            end = min(start + size, len(normalized))
            if end < len(normalized):
                boundary = normalized.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(normalized):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def __init__(self, ollama, repository, audit, embedding_batch_size=32, max_upload_bytes=30 * 1024 * 1024, max_text_bytes=50 * 1024 * 1024):
        """ Handle the   init   operation. """
        self.ollama = ollama
        self.repository = repository
        self.audit = audit
        self.embedding_batch_size = max(1, embedding_batch_size)
        self.max_upload_bytes = max_upload_bytes
        self.max_text_bytes = max_text_bytes

    def ingest(self, collection, title, chunks, tags, embedding_model):
        """ Embed document chunks and persist them under a unique document ID. """
        if not collection or not collection.strip():
            raise ValidationError("Collection name is required.")
        if not title or not title.strip():
            raise ValidationError("Document title is required.")
        if not chunks:
            raise ValidationError("No chunks supplied.")
        if not embedding_model:
            raise ValidationError("Embedding model is required.")

        document_id = str(uuid.uuid4())
        vectors = []
        try:
            for offset in range(0, len(chunks), self.embedding_batch_size):
                batch = chunks[offset:offset + self.embedding_batch_size]
                vectors.extend(self.ollama.generate_embeddings(embedding_model, batch))
            self.repository.store_document(
                collection=collection, document_id=document_id, title=title,
                chunks=chunks, vectors=vectors, tags=tags,
                embedding_model=embedding_model
            )
            return {"document_id": document_id, "chunk_count": len(chunks)}
        except Exception as exc:
            self.audit.record_event(
                event_type="INGESTION_FAILURE",
                status="FAILED",
                details={"collection": collection, "document_id": document_id,
                         "title": title, "error": str(exc)}
            )
            raise
