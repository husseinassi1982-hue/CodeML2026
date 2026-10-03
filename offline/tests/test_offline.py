"""Run with:  python -m unittest discover -s tests -v   (from the dayone/ folder)"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptography.fernet import Fernet

from offline import (AccessDenied, Connectivity, IllegalTransition, RecordStore, State,
                     process_queue, retry_failed_processing, sync_saved, retry_failed_sync)


def fake_extract(image_bytes, record_id):
    return {"age": {"value": 31, "status": "KNOWN", "confidence": 0.9}}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.key = Fernet.generate_key()
        self.store = RecordStore(self.tmp.name, key=self.key)
        self.net = Connectivity(online=False)

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def photo(self, n):
        return f"fake-photo-{n}".encode()


class TestQueue(Base):
    def test_offline_capture_then_online_drains_everything(self):
        ids = [self.store.capture(self.photo(i), "mw-1") for i in range(5)]
        report = process_queue(self.store, self.net, fake_extract)       # still offline
        self.assertEqual(len(report["waiting"]), 5)
        self.assertTrue(all(self.store.get(i)["state"] == State.PENDING_AI for i in ids))

        self.net.set_online(True)
        report = process_queue(self.store, self.net, fake_extract)
        self.assertEqual(report["processed"], ids)                       # oldest first
        self.assertTrue(all(self.store.get(i)["state"] == State.TO_REVIEW for i in ids))
        self.assertEqual(self.store.get(ids[0])["fields"]["age"]["value"], 31)

    def test_connection_drops_midway_nothing_lost(self):
        ids = [self.store.capture(self.photo(i), "mw-1") for i in range(3)]
        self.net.set_online(True)
        calls = []

        def flaky(image, rid):
            calls.append(rid)
            if len(calls) == 2:                 # internet dies on the 2nd photo
                self.net.set_online(False)
                raise ConnectionError("lost connection")
            return fake_extract(image, rid)

        report = process_queue(self.store, self.net, flaky)
        self.assertEqual(report["processed"], [ids[0]])
        self.assertEqual(self.store.get(ids[1])["state"], State.PENDING_AI)
        self.assertEqual(self.store.get(ids[2])["state"], State.PENDING_AI)

        self.net.set_online(True)
        report = process_queue(self.store, self.net, fake_extract)
        self.assertEqual(report["processed"], [ids[1], ids[2]])

    def test_ai_failure_retries_then_gives_up_then_can_retry(self):
        rid = self.store.capture(self.photo(1), "mw-1")
        self.net.set_online(True)

        def broken(image, record_id):
            raise ValueError("model crashed")

        for _ in range(2):
            process_queue(self.store, self.net, broken, max_attempts=3)
            self.assertEqual(self.store.get(rid)["state"], State.PENDING_AI)
        process_queue(self.store, self.net, broken, max_attempts=3)
        self.assertEqual(self.store.get(rid)["state"], State.PROCESSING_FAILED)

        retry_failed_processing(self.store)
        self.assertEqual(self.store.get(rid)["state"], State.PENDING_AI)
        process_queue(self.store, self.net, fake_extract)
        self.assertEqual(self.store.get(rid)["state"], State.TO_REVIEW)

    def test_restart_keeps_the_queue(self):
        rid = self.store.capture(self.photo(1), "mw-1")
        self.store.db.close()
        reopened = RecordStore(self.tmp.name, key=self.key)   # "app restarted"
        self.assertEqual(reopened.get(rid)["state"], State.PENDING_AI)
        self.net.set_online(True)
        process_queue(reopened, self.net, fake_extract)
        self.assertEqual(reopened.get(rid)["state"], State.TO_REVIEW)
        self.store = reopened

    def test_duplicate_photo_is_parked_not_queued(self):
        first = self.store.capture(self.photo(1), "mw-1")
        second = self.store.capture(self.photo(1), "mw-1")
        self.assertEqual(self.store.get(first)["state"], State.PENDING_AI)
        self.assertEqual(self.store.get(second)["state"], State.SUSPECTED_DUPLICATE)


class TestStates(Base):
    def test_illegal_transition_rejected_and_state_unchanged(self):
        rid = self.store.capture(self.photo(1), "mw-1")
        with self.assertRaises(IllegalTransition):
            self.store.transition(rid, State.SYNCED)
        self.assertEqual(self.store.get(rid)["state"], State.PENDING_AI)

    def test_full_happy_path_and_sync_failure_recovery(self):
        rid = self.store.capture(self.photo(1), "mw-1")
        self.net.set_online(True)
        process_queue(self.store, self.net, fake_extract)
        for s in (State.VALIDATED, State.PATIENT_LINKED, State.SAVED):
            self.store.transition(rid, s)

        def rejecting_server(rec):
            raise ValueError("server said no")

        sync_saved(self.store, self.net, rejecting_server)
        self.assertEqual(self.store.get(rid)["state"], State.SYNC_FAILED)
        retry_failed_sync(self.store)
        sync_saved(self.store, self.net, lambda rec: None)
        self.assertEqual(self.store.get(rid)["state"], State.SYNCED)
        states = [h["new_state"] for h in self.store.history(rid)]
        self.assertEqual(states[0], "CAPTURED")
        self.assertEqual(states[-1], "SYNCED")


class TestPrivacy(Base):
    def test_nothing_readable_on_disk(self):
        marker = b"VERY-RECOGNISABLE-PHOTO-BYTES"
        rid = self.store.capture(marker, "mw-1")
        self.store.update_fields(rid, {"note": "SECRET-FIELD-VALUE"})
        self.store.db.execute("PRAGMA wal_checkpoint(FULL)")
        for path in Path(self.tmp.name).rglob("*"):
            if path.is_file() and path.name != ".key":
                data = path.read_bytes()
                self.assertNotIn(marker, data, path)
                self.assertNotIn(b"SECRET-FIELD-VALUE", data, path)

    def test_original_image_roundtrip_and_role_check(self):
        photo = self.photo(7)
        rid = self.store.capture(photo, "mw-1")
        self.assertEqual(self.store.get_image(rid, "midwife"), photo)   # unchanged bytes
        with self.assertRaises(AccessDenied):
            self.store.get_image(rid, "researcher")

    def test_wrong_key_cannot_read(self):
        rid = self.store.capture(self.photo(1), "mw-1")
        self.store.update_fields(rid, {"a": 1})
        other = RecordStore(self.tmp.name, key=Fernet.generate_key())
        with self.assertRaises(Exception):
            other.get(rid)
        other.db.close()

    def test_record_id_is_random_uuid(self):
        a = self.store.capture(self.photo(1), "mw-1")
        b = self.store.capture(self.photo(2), "mw-1")
        self.assertNotEqual(a, b)
        self.assertEqual(len(a), 36)


if __name__ == "__main__":
    unittest.main()
