""" Semantic retrieval and retrieval-augmented answer generation. """

import time
import uuid
from services.ollama_service import OllamaError
from exceptions.app_exceptions import RetrievalError, ValidationError

class RetrievalService:
    """ RetrievalService groups related application behavior. """
    def __init__(self, ollama, repository, audit):
        """ Handle the   init   operation. """
        self.ollama = ollama
        self.repository = repository
        self.audit = audit

    def retrieve(self, collection, question, embedding_model, top_k=5):
        """ Embed a question and return the most relevant source chunks. """
        if not question or not question.strip():
            raise ValidationError("Question cannot be empty.")
        if top_k < 1 or top_k > 50:
            raise ValidationError("top_k must be between 1 and 50.")
        try:
            contract = self.audit.get_collection_contract(collection)
            if contract is None:
                raise ValidationError(
                    f"Collection '{collection}' has no registered embedding contract. "
                    "Ingest it through this application or perform an explicit migration."
                )
            if contract["embedding_model"] != embedding_model:
                raise ValidationError(
                    f"Collection '{collection}' is locked to embedding model "
                    f"'{contract['embedding_model']}', not '{embedding_model}'. "
                    "Select the registered model or re-embed into a new collection."
                )
            vector = self.ollama.generate_embeddings(
                embedding_model, ["search_query: " + question]
            )[0]
            points = self.repository.search(collection, vector, top_k)
            sources = []
            for point in points:
                payload = point.payload or {}
                sources.append({
                    "title": payload.get("title", "Unknown"),
                    "document_id": payload.get("document_id"),
                    "chunk_index": payload.get("chunk_index", -1),
                    "score": float(point.score or 0),
                    "content": payload.get("content", "")
                })
            return sources
        except Exception as exc:
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

class RAGService:
    """ RAGService groups related application behavior. """
    def __init__(self, ollama, retrieval, audit, default_embedding_model=None):
        """ Handle the   init   operation. """
        self.ollama = ollama
        self.retrieval = retrieval
        self.audit = audit
        self.default_embedding_model = default_embedding_model

    def answer(self, collection, question, chat_model, top_k=5, embedding_model=None):
        """ Build a grounded prompt, invoke the chat model, and audit the result. """
        request_id = str(uuid.uuid4())
        started = time.monotonic()
        sources, context, answer = [], "", ""
        model_for_query = embedding_model or self.default_embedding_model
        try:
            if not model_for_query:
                raise ValidationError("No embedding model configured for retrieval.")
            sources = self.retrieval.retrieve(
                collection, question, model_for_query, top_k
            )
            if not sources:
                answer = "I could not find relevant information in the selected collection."
                return answer, sources, {
                    "request_id": request_id,
                    "duration_seconds": time.monotonic() - started
                }
            context = "\n\n".join(
                f"[SOURCE {i}]\nDocument: {s['title']}\nChunk: {s['chunk_index']}\n{s['content']}"
                for i, s in enumerate(sources, 1)
            )
            messages = [
                {"role": "system", "content":
                    "Answer using the supplied context. Do not invent facts. "
                    "If the context does not contain the answer, say so. "
                    "Cite supporting material as [SOURCE 1], [SOURCE 2], etc.\n\n"
                    f"CONTEXT:\n{context}"},
                {"role": "user", "content": question}
            ]
            result = self.ollama.chat(chat_model, messages)
            answer = result["content"]
            duration = time.monotonic() - started
            self.audit.record_inference(
                request_id=request_id, model=chat_model, collection=collection,
                question=question, sources=sources, context_sent=context,
                response=answer, duration_seconds=duration, status="SUCCESS",
                error=None, prompt_tokens=result.get("prompt_tokens"),
                generated_tokens=result.get("generated_tokens")
            )
            return answer, sources, {"request_id": request_id, "duration_seconds": duration}
        except Exception as exc:
            duration = time.monotonic() - started
            self.audit.record_inference(
                request_id=request_id, model=chat_model, collection=collection,
                question=question, sources=sources, context_sent=context,
                response=answer, duration_seconds=duration, status="FAILED",
                error=str(exc)
            )
            raise
