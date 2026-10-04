"""Shared state for the running app: the record store, patients, and the network switch.

Records (state machine, encrypted fields and photos) live in offlineModule's RecordStore.
Patients are new: a small table in its own SQLite file next to the records.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from core import config  # noqa: F401  (sets sys.path)
from offline import Connectivity, RecordStore


def _now() -> str:
    """Current time as UTC ISO 8601 text."""
    return datetime.now(timezone.utc).isoformat()


class PatientStore:
    """Patient profiles keyed by the code the midwife writes on the paper registry.

    Only the code is stored: no name, no ID number, nothing identifying."""

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(root / "patients.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self.db:
            self.db.execute("""CREATE TABLE IF NOT EXISTS patients (
                code TEXT PRIMARY KEY, created_at TEXT NOT NULL)""")
            self.db.execute("""CREATE TABLE IF NOT EXISTS links (
                record_id TEXT PRIMARY KEY, patient_code TEXT NOT NULL, linked_at TEXT NOT NULL)""")

    def exists(self, code: str) -> bool:
        """Is there a profile with this code?"""
        with self._lock:
            return self.db.execute("SELECT 1 FROM patients WHERE code = ?", (code,)).fetchone() is not None

    def create(self, code: str) -> None:
        """Add a profile. The code must be new (the primary key refuses duplicates)."""
        with self._lock, self.db:
            self.db.execute("INSERT INTO patients (code, created_at) VALUES (?, ?)", (code, _now()))

    def all_codes(self) -> list[str]:
        """Every profile code, oldest first."""
        with self._lock:
            return [r["code"] for r in self.db.execute("SELECT code FROM patients ORDER BY created_at")]

    def link(self, record_id: str, code: str) -> None:
        """Attach a record to a profile; linking the same record again replaces its link."""
        with self._lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO links (record_id, patient_code, linked_at) VALUES (?,?,?)",
                            (record_id, code, _now()))

    def code_for_record(self, record_id: str) -> str | None:
        """Code of the profile this record is linked to, or None."""
        with self._lock:
            row = self.db.execute("SELECT patient_code FROM links WHERE record_id = ?", (record_id,)).fetchone()
        return row["patient_code"] if row else None

    def records_for(self, code: str) -> list[str]:
        """IDs of a profile's records, in the order they were linked."""
        with self._lock:
            rows = self.db.execute("SELECT record_id FROM links WHERE patient_code = ? ORDER BY linked_at",
                                   (code,)).fetchall()
        return [r["record_id"] for r in rows]


class AppState:
    """Everything the running app shares: the record store, the patients, the network switch, the AI
    queue lock and the pretend health-system server."""

    def __init__(self, root: Path):
        self.store = RecordStore(root)
        self.patients = PatientStore(root)
        self.net = Connectivity(online=True)
        self.queue_lock = threading.Lock()  # one queue run at a time
        self.server_received: list[str] = []  # the pretend health-system server (demo)


_state: AppState | None = None


def get_state() -> AppState:
    """The app's shared state, created in config.DATA_DIR on first use."""
    global _state
    if _state is None:
        _state = AppState(config.DATA_DIR)
    return _state


def reset_state(root: Path) -> AppState:
    """Tests and the demo reset button use a fresh data folder."""
    global _state
    _state = AppState(root)
    return _state
