""" Qdrant persistence adapter for collections, vectors, documents, and points. """

import uuid
from qdrant_client import QdrantClient, models

class QdrantRepository:
    """ QdrantRepository groups related application behavior. """
    def __init__(self, url, audit):
        """ Handle the   init   operation. """
        self.client = QdrantClient(url=url)
        self.audit = audit

    def health_check(self):
        """ Check dependency availability and return a safe status result. """
        self.client.get_collections()
        return True

    def list_collections(self):
        """ Handle the list collections operation. """
        return [c.name for c in self.client.get_collections().collections]

    def ensure_collection(self, collection, dimension, embedding_model):
        """ Create a Qdrant collection if needed and enforce its embedding contract. """
        collection_is_new = collection not in self.list_collections()
        if collection_is_new:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=dimension, distance=models.Distance.COSINE
                )
            )
        info = self.client.get_collection(collection)

        # A vector dimension does not identify the embedding model or vector space.
        # Refuse to guess the model for an existing populated collection.
        existing_contract = self.audit.get_collection_contract(collection)
        if not collection_is_new and existing_contract is None and (info.points_count or 0) > 0:
            raise ValueError(
                f"Collection '{collection}' contains vectors but has no registered "
                "embedding contract. Rebuild it into a new collection or migrate it explicitly."
            )
        configured = info.config.params.vectors
        if isinstance(configured, dict):
            raise ValueError("Named vectors are not supported by this repository.")
        if configured.size != dimension:
            raise ValueError(
                f"Qdrant dimension mismatch: collection={configured.size}, incoming={dimension}."
            )
        # SQLite registry is the authoritative contract, including the model identity.
        self.audit.register_collection_contract(
            collection, embedding_model, dimension, "Cosine"
        )

    def store_document(self, collection, document_id, title, chunks, vectors, tags, embedding_model):
        """ Write prepared vectors and create a manifest for crash recovery. """
        if len(chunks) != len(vectors) or not vectors:
            raise ValueError("Chunk/vector count mismatch or empty vectors.")
        dimension = len(vectors[0])
        if any(len(v) != dimension for v in vectors):
            raise ValueError("Inconsistent vector dimensions.")

        self.ensure_collection(collection, dimension, embedding_model)

        point_ids = [str(uuid.uuid4()) for _ in chunks]
        points = [
            models.PointStruct(
                id=point_ids[index],
                vector=vector,
                payload={
                    "document_id": document_id,
                    "title": title,
                    "content": chunk,
                    "chunk_index": index,
                    "tags": tags,
                    "embedding_model": embedding_model,
                },
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]

        self.audit.create_ingestion_manifest(
            document_id, collection, title, point_ids
        )
        try:
            # One upsert request after all chunks and vectors have been prepared.
            self.client.upsert(collection_name=collection, points=points, wait=True)
            self.audit.update_ingestion_manifest(document_id, "COMMITTED")
        except Exception as exc:
            # Compensating rollback: remove every point ID in this ingestion batch.
            rollback_error = None
            try:
                self.client.delete(
                    collection_name=collection,
                    points_selector=models.PointIdsList(points=point_ids),
                    wait=True,
                )
            except Exception as rollback_exc:
                rollback_error = str(rollback_exc)
            status = "ROLLBACK_FAILED" if rollback_error else "ROLLED_BACK"
            self.audit.update_ingestion_manifest(
                document_id, status,
                error=f"Upsert error: {exc}; rollback error: {rollback_error}" if rollback_error else str(exc)
            )
            self.audit.record_event(
                "INGESTION_ROLLBACK", status,
                {"document_id": document_id, "collection": collection,
                 "point_count": len(point_ids), "error": str(exc),
                 "rollback_error": rollback_error}
            )
            if rollback_error:
                raise RuntimeError(
                    f"Qdrant upsert failed ({exc}); rollback also failed ({rollback_error}). "
                    f"Reconciliation required for document {document_id}."
                ) from exc
            raise


    def get_points_by_ids(self, collection, point_ids):
        """ Retrieve exact point IDs to reconcile an interrupted write. """
        if not point_ids:
            return []
        return self.client.retrieve(
            collection_name=collection,
            ids=point_ids,
            with_payload=True,
            with_vectors=False
        )

    def delete_points_by_ids(self, collection, point_ids):
        """ Delete only the point IDs recorded in a manifest. """
        if not point_ids:
            return
        self.client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=point_ids),
            wait=True
        )

    def collection_exists(self, collection):
        """ Check whether a Qdrant collection exists. """
        return collection in self.list_collections()

    def search(self, collection, vector, limit=5):
        """ Find the nearest Qdrant vectors for a query embedding. """
        result = self.client.query_points(
            collection_name=collection, query=vector, limit=limit,
            with_payload=True
        )
        return result.points

    def list_documents(self, collection):
        """ Aggregate vector points into document-level summaries. """
        points, _ = self.client.scroll(
            collection_name=collection, limit=10000,
            with_payload=True, with_vectors=False
        )
        docs = {}
        for point in points:
            p = point.payload or {}
            doc_id = p.get("document_id")
            if not doc_id:
                continue
            if doc_id not in docs:
                docs[doc_id] = {"title": p.get("title", "Untitled"), "chunks": 0, "tags": p.get("tags", [])}
            docs[doc_id]["chunks"] += 1
        return docs

    def delete_document(self, collection, document_id):
        """ Delete all points associated with one document ID. """
        self.client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ])
            ),
            wait=True
        )

    def close(self):
        """ Release the HTTP client's resources. """
        self.client.close()
