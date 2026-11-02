""" Unit tests for interrupted ingestion recovery scenarios. """

from dataclasses import dataclass
from services.recovery_service import RecoveryService

@dataclass
class FakePoint:
    """ FakePoint groups related application behavior. """
    id: str
    payload: dict

class FakeAudit:
    """ FakeAudit groups related application behavior. """
    def __init__(self, manifests):
        """ Handle the   init   operation. """
        self.manifests = manifests
        self.statuses = {}
        self.events = []

    def claim_stale_ingestion_manifests(self, stale_seconds=120, limit=100):
        """ Atomically claim old unfinished manifests for recovery. """
        claimed = self.manifests[:limit]
        self.manifests = []
        return claimed

    def update_ingestion_manifest(self, document_id, status, error=None):
        """ Update the durable state of an ingestion operation. """
        self.statuses[document_id] = (status, error)

    def record_event(self, event_type, status, details):
        """ Persist an operational event for troubleshooting and audit. """
        self.events.append((event_type, status, details))

class FakeRepository:
    """ FakeRepository groups related application behavior. """
    def __init__(self, points=None, collection_exists=True, fail_delete=False):
        """ Handle the   init   operation. """
        self.points = points or {}
        self.exists = collection_exists
        self.fail_delete = fail_delete

    def collection_exists(self, collection):
        """ Check whether a Qdrant collection exists. """
        return self.exists

    def get_points_by_ids(self, collection, point_ids):
        """ Retrieve exact point IDs to reconcile an interrupted write. """
        return [self.points[p] for p in point_ids if p in self.points]

    def delete_points_by_ids(self, collection, point_ids):
        """ Delete only the point IDs recorded in a manifest. """
        if self.fail_delete:
            raise RuntimeError("simulated delete failure")
        for point_id in point_ids:
            self.points.pop(point_id, None)

def manifest(doc_id, ids):
    """ Handle the manifest operation. """
    return {
        "document_id": doc_id,
        "collection_name": "docs",
        "title": "example",
        "point_ids": ids,
        "status": "RECOVERING",
    }

def test_recovery_marks_fully_written_document_committed():
    """ Handle the test recovery marks fully written document committed operation. """
    ids = ["p1", "p2"]
    audit = FakeAudit([manifest("d1", ids)])
    repo = FakeRepository({
        "p1": FakePoint("p1", {"document_id": "d1"}),
        "p2": FakePoint("p2", {"document_id": "d1"}),
    })
    report = RecoveryService(audit, repo).run(stale_after_seconds=0)
    assert report["committed"] == 1
    assert audit.statuses["d1"][0] == "COMMITTED"

def test_recovery_rolls_back_partial_write():
    """ Handle the test recovery rolls back partial write operation. """
    ids = ["p1", "p2"]
    audit = FakeAudit([manifest("d2", ids)])
    repo = FakeRepository({"p1": FakePoint("p1", {"document_id": "d2"})})
    report = RecoveryService(audit, repo).run(stale_after_seconds=0)
    assert report["rolled_back"] == 1
    assert repo.points == {}
    assert audit.statuses["d2"][0] == "ROLLED_BACK"

def test_recovery_records_failed_rollback_for_retry():
    """ Handle the test recovery records failed rollback for retry operation. """
    ids = ["p1", "p2"]
    audit = FakeAudit([manifest("d3", ids)])
    repo = FakeRepository(
        {"p1": FakePoint("p1", {"document_id": "d3"})},
        fail_delete=True
    )
    report = RecoveryService(audit, repo).run(stale_after_seconds=0)
    assert report["failed"] == 1
    assert audit.statuses["d3"][0] == "ROLLBACK_FAILED"

def test_missing_collection_is_treated_as_rolled_back():
    """ Handle the test missing collection is treated as rolled back operation. """
    audit = FakeAudit([manifest("d4", ["p1"])])
    repo = FakeRepository(collection_exists=False)
    report = RecoveryService(audit, repo).run(stale_after_seconds=0)
    assert report["rolled_back"] == 1
