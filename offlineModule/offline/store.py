"""Encrypted local storage for records and their original photos.

- Metadata (state, timestamps, ...) lives in SQLite.
- The extracted fields (JSON) and the photo bytes are encrypted with Fernet
  before they touch the disk.
- Record ids are random UUIDs: never derived from anything personal.
- The original photo is stored byte-for-byte (encrypted) and never modified;
  its SHA-256 is kept so we can spot the same photo being captured twice.

DEMO NOTE: the key is kept in a local file / env var. On a real phone it would
live in the OS keystore (Android Keystore / iOS Keychain).
"""
import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cryptography.fernet import Fernet

from .states import State, check_transition

# Roles allowed to see the original photo (privacy requirement: restricted access)
IMAGE_ROLES = {"midwife", "admin"}


class AccessDenied(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_key(root: Path) -> bytes:
    """Key from DAYONE_KEY env var, else from root/.key (created if missing)."""
    env = os.environ.get("DAYONE_KEY")
    if env:
        return env.encode()
    key_file = root / ".key"
    if not key_file.exists():
        key_file.write_bytes(Fernet.generate_key())
        os.chmod(key_file, 0o600)
    return key_file.read_bytes()


class RecordStore:
    def __init__(self, root="data", key: bytes | None = None):
        self.root = Path(root)
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        self.fernet = Fernet(key or load_key(self.root))
        self.db = sqlite3.connect(self.root / "records.db")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")  # safer if the app dies mid-write
        self._create_tables()

    def _create_tables(self):
        with self.db:
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    midwife_id TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    image_sha256 TEXT NOT NULL,
                    fields_enc BLOB,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT
                )""")
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL,
                    at TEXT NOT NULL,
                    old_state TEXT,
                    new_state TEXT NOT NULL,
                    note TEXT
                )""")

    # ---------- capture ----------
    def capture(self, image_bytes: bytes, midwife_id: str) -> str:
        """Save the photo + a new record. Returns the record id.

        Order matters for crash safety: image file first (written to a temp
        name, then renamed), database row second. If the row insert fails we
        delete the file, so we never keep half a record.
        """
        record_id = str(uuid.uuid4())
        sha = hashlib.sha256(image_bytes).hexdigest()
        duplicate = self.db.execute(
            "SELECT 1 FROM records WHERE image_sha256 = ? LIMIT 1", (sha,)
        ).fetchone() is not None

        path = self._image_path(record_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(self.fernet.encrypt(image_bytes))
        os.replace(tmp, path)  # atomic rename
        try:
            now = _now()
            with self.db:
                self.db.execute(
                    "INSERT INTO records (id, state, midwife_id, captured_at, updated_at,"
                    " image_sha256) VALUES (?,?,?,?,?,?)",
                    (record_id, State.CAPTURED.value, midwife_id, now, now, sha),
                )
                self.db.execute(
                    "INSERT INTO history (record_id, at, old_state, new_state, note)"
                    " VALUES (?,?,NULL,?,?)",
                    (record_id, now, State.CAPTURED.value, "photo captured"),
                )
        except Exception:
            path.unlink(missing_ok=True)
            raise

        # Same photo seen before -> park it for a human, otherwise queue it.
        if duplicate:
            self.transition(record_id, State.SUSPECTED_DUPLICATE, note="same image hash seen before")
        else:
            self.transition(record_id, State.PENDING_AI, note="queued")
        return record_id

    # ---------- reading ----------
    def get(self, record_id: str) -> dict:
        row = self.db.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
        if row is None:
            raise KeyError(record_id)
        rec = dict(row)
        enc = rec.pop("fields_enc")
        rec["fields"] = json.loads(self.fernet.decrypt(enc)) if enc else {}
        rec["state"] = State(rec["state"])
        return rec

    def list_by_state(self, *states: State) -> list[dict]:
        marks = ",".join("?" for _ in states)
        rows = self.db.execute(
            f"SELECT id FROM records WHERE state IN ({marks}) ORDER BY captured_at, rowid",
            [s.value for s in states],
        ).fetchall()
        return [self.get(r["id"]) for r in rows]

    def history(self, record_id: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT at, old_state, new_state, note FROM history WHERE record_id = ? ORDER BY id",
            (record_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- changing ----------
    def transition(self, record_id: str, new_state: State, note: str | None = None,
                   error: str | None = None) -> None:
        """The ONLY way to change a record's state. Illegal moves raise."""
        with self.db:
            row = self.db.execute("SELECT state FROM records WHERE id = ?", (record_id,)).fetchone()
            if row is None:
                raise KeyError(record_id)
            old = State(row["state"])
            check_transition(old, new_state)
            now = _now()
            self.db.execute(
                "UPDATE records SET state = ?, updated_at = ?, last_error = ? WHERE id = ?",
                (new_state.value, now, error, record_id),
            )
            self.db.execute(
                "INSERT INTO history (record_id, at, old_state, new_state, note) VALUES (?,?,?,?,?)",
                (record_id, now, old.value, new_state.value, note or error),
            )

    def update_fields(self, record_id: str, fields: dict) -> None:
        """Merge fields (from the AI or from the midwife's corrections) into the record."""
        current = self.get(record_id)["fields"]
        current.update(fields)
        enc = self.fernet.encrypt(json.dumps(current, ensure_ascii=False).encode())
        with self.db:
            self.db.execute(
                "UPDATE records SET fields_enc = ?, updated_at = ? WHERE id = ?",
                (enc, _now(), record_id),
            )

    def bump_attempts(self, record_id: str) -> int:
        with self.db:
            self.db.execute("UPDATE records SET attempts = attempts + 1 WHERE id = ?", (record_id,))
        return self.get(record_id)["attempts"]

    def reset_attempts(self, record_id: str) -> None:
        with self.db:
            self.db.execute("UPDATE records SET attempts = 0 WHERE id = ?", (record_id,))

    # ---------- images ----------
    def _image_path(self, record_id: str) -> Path:
        return self.root / "images" / f"{record_id}.enc"

    def load_image_for_processing(self, record_id: str) -> bytes:
        """For the pipeline itself (extraction). Not exposed to end users."""
        return self.fernet.decrypt(self._image_path(record_id).read_bytes())

    def get_image(self, record_id: str, role: str) -> bytes:
        """For anyone asking on behalf of a user: role-checked."""
        if role not in IMAGE_ROLES:
            raise AccessDenied(f"role '{role}' may not view original images")
        return self.load_image_for_processing(record_id)
