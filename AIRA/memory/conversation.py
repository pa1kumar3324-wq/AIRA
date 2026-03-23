"""
memory/conversation.py

Conversation memory manager for AIRA.
Handles in-session message history and persistent storage of past sessions.

Each session is saved as a JSON file under memory/sessions/.
Sessions are indexed in memory/index.json for quick lookup.

Message format:
    {
        "role":      "user" | "assistant",
        "content":   str,
        "timestamp": ISO 8601 str,
        "emotion":   str | None      # only on user messages
    }
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

SESSIONS_DIR   = Path(__file__).parent / "sessions"
INDEX_FILE     = Path(__file__).parent / "index.json"
MAX_CONTEXT    = 20   # max messages kept in active context window


class ConversationMemory:
    def __init__(self, username: str = "User"):
        self.username    = username
        self.session_id  = str(uuid.uuid4())[:8]
        self.started_at  = datetime.now().isoformat()
        self.messages: list[dict] = []

        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Active session ──────────────────────────────────────────────────────

    def add_user_message(self, content: str, emotion: str = None):
        """Record a user message with optional emotion label."""
        self.messages.append({
            "role":      "user",
            "content":   content,
            "timestamp": datetime.now().isoformat(),
            "emotion":   emotion,
        })

    def add_assistant_message(self, content: str):
        """Record AIRA's response."""
        self.messages.append({
            "role":      "assistant",
            "content":   content,
            "timestamp": datetime.now().isoformat(),
            "emotion":   None,
        })

    def get_context(self) -> list[dict]:
        """
        Return the last MAX_CONTEXT messages in the format Ollama expects.
        Strips out metadata fields (timestamp, emotion) — only role + content.
        """
        trimmed = self.messages[-MAX_CONTEXT:]
        return [{"role": m["role"], "content": m["content"]} for m in trimmed]

    def clear_session(self):
        """Wipe the current in-memory session (does not affect saved sessions)."""
        self.messages = []
        self.session_id = str(uuid.uuid4())[:8]
        self.started_at = datetime.now().isoformat()

    def message_count(self) -> int:
        return len(self.messages)

    def last_emotion(self) -> str | None:
        for msg in reversed(self.messages):
            if msg["role"] == "user" and msg.get("emotion"):
                return msg["emotion"]
        return None

    # ── Persistence ─────────────────────────────────────────────────────────

    def save_session(self):
        """Save the current session to disk."""
        if not self.messages:
            return

        session_data = {
            "session_id":  self.session_id,
            "username":    self.username,
            "started_at":  self.started_at,
            "ended_at":    datetime.now().isoformat(),
            "message_count": len(self.messages),
            "messages":    self.messages,
        }

        session_file = SESSIONS_DIR / f"{self.session_id}.json"
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        self._update_index(session_data)

    def _update_index(self, session_data: dict):
        """Add or update the session entry in the index file."""
        index = self._load_index()
        index[self.session_id] = {
            "session_id":    session_data["session_id"],
            "username":      session_data["username"],
            "started_at":    session_data["started_at"],
            "ended_at":      session_data["ended_at"],
            "message_count": session_data["message_count"],
        }
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

    def _load_index(self) -> dict:
        if INDEX_FILE.exists():
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    # ── History retrieval ────────────────────────────────────────────────────

    def get_all_sessions(self) -> list[dict]:
        """Return a list of all past sessions (summary only, sorted newest first)."""
        index = self._load_index()
        sessions = list(index.values())
        sessions.sort(key=lambda s: s["started_at"], reverse=True)
        return sessions

    def get_session(self, session_id: str) -> dict | None:
        """Load the full message log for a specific session."""
        session_file = SESSIONS_DIR / f"{session_id}.json"
        if not session_file.exists():
            return None
        with open(session_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def clear_all_history(self):
        """Delete all saved sessions and the index."""
        if SESSIONS_DIR.exists():
            for f in SESSIONS_DIR.glob("*.json"):
                f.unlink()
        if INDEX_FILE.exists():
            INDEX_FILE.unlink()
