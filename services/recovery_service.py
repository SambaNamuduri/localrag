""" Reconciliation of interrupted Qdrant ingestion operations. """

import json
import logging

logger = logging.getLogger(__name__)

class RecoveryService:
    """
    Reconciles ingestion manifests with Qdrant after an interrupted process.

    Policy:
    - Every expected point exists and belongs to the manifest document:
      mark COMMITTED (upsert completed before the crash).
    - Some or none of the expected points exist:
      delete the complete expected ID set, verify absence, mark ROLLED_BACK.
    - Any uncertainty or failed verification:
      leave a retryable ROLLBACK_FAILED status.
    """

    def __init__(self, audit, repository):
        """ Handle the   init   operation. """
        self.audit = audit
        self.repository = repository

    def run(self, stale_after_seconds=120, limit=100):
        """ Reconcile stale ingestion manifests against Qdrant and verify outcomes. """
        if stale_after_seconds < 0:
            raise ValueError("stale_after_seconds cannot be negative.")
        manifests = self.audit.claim_stale_ingestion_manifests(
            stale_seconds=stale_after_seconds,
            limit=limit
        )
        report = {
            "claimed": len(manifests),
            "committed": 0,
            "rolled_back": 0,
            "failed": 0,
            "errors": []
        }

        for manifest in manifests:
            document_id = manifest["document_id"]
            collection = manifest["collection_name"]
            point_ids = manifest["point_ids"]

            try:
                if not point_ids:
                    self.audit.update_ingestion_manifest(
                        document_id, "ROLLED_BACK",
                        error="Manifest had no point IDs; no Qdrant points could be committed."
                    )
                    report["rolled_back"] += 1
                    continue

                if not self.repository.collection_exists(collection):
                    self.audit.update_ingestion_manifest(
                        document_id, "ROLLED_BACK",
                        error="Qdrant collection does not exist; no points remain."
                    )
                    report["rolled_back"] += 1
                    continue

                found = self.repository.get_points_by_ids(collection, point_ids)
                found_by_id = {str(point.id): point for point in found}

                all_present_and_owned = len(found_by_id) == len(set(point_ids))
                if all_present_and_owned:
                    for point in found_by_id.values():
                        payload = point.payload or {}
                        if payload.get("document_id") != document_id:
                            all_present_and_owned = False
                            break

                if all_present_and_owned:
                    self.audit.update_ingestion_manifest(
                        document_id, "COMMITTED",
                        error="Recovered: all expected Qdrant points were verified."
                    )
                    self.audit.record_event(
                        "INGESTION_RECOVERY", "COMMITTED",
                        {"document_id": document_id, "collection": collection,
                         "verified_points": len(point_ids)}
                    )
                    report["committed"] += 1
                    continue

                # Partial upsert or mismatched payload: remove only manifest-owned IDs.
                self.repository.delete_points_by_ids(collection, point_ids)
                remaining = self.repository.get_points_by_ids(collection, point_ids)

                if remaining:
                    raise RuntimeError(
                        f"Rollback verification failed: {len(remaining)} point(s) remain."
                    )

                self.audit.update_ingestion_manifest(
                    document_id, "ROLLED_BACK",
                    error=(
                        "Recovered: partial or inconsistent Qdrant write was deleted "
                        "and absence of all expected point IDs was verified."
                    )
                )
                self.audit.record_event(
                    "INGESTION_RECOVERY", "ROLLED_BACK",
                    {"document_id": document_id, "collection": collection,
                     "expected_points": len(point_ids),
                     "found_before_rollback": len(found_by_id)}
                )
                report["rolled_back"] += 1

            except Exception as exc:
                message = str(exc)
                logger.exception(
                    "Recovery failed for document_id=%s collection=%s",
                    document_id, collection
                )
                try:
                    self.audit.update_ingestion_manifest(
                        document_id, "ROLLBACK_FAILED", error=message
                    )
                    self.audit.record_event(
                        "INGESTION_RECOVERY", "FAILED",
                        {"document_id": document_id, "collection": collection,
                         "error": message}
                    )
                except Exception:
                    logger.exception("Could not persist recovery failure state.")
                report["failed"] += 1
                report["errors"].append({
                    "document_id": document_id,
                    "collection": collection,
                    "error": message
                })

        self.audit.record_event("INGESTION_RECOVERY_RUN", "COMPLETED", report)
        return report
