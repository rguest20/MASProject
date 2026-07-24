"""Lazy access to the optional human-language dictionary.

The source file is intentionally large.  It is read once only when a human
actually introduces a word, and only the queried entry plus a few related
tokens are handed to agents.
"""

from __future__ import annotations

import json
from pathlib import Path
import re


class HumanDictionary:
    def __init__(self, path=None):
        self.path = Path(path) if path else Path(__file__).resolve().parents[1] / "filtered.json"
        self._entries = None
        self.available = None

    def _load(self):
        if self._entries is not None:
            return self._entries
        try:
            with self.path.open(encoding="utf-8") as source:
                entries = json.load(source)
            self._entries = entries if isinstance(entries, dict) else {}
            self.available = True
        except (OSError, json.JSONDecodeError):
            self._entries = {}
            self.available = False
        return self._entries

    def entry(self, token):
        if not isinstance(token, str) or not token:
            return None
        entries = self._load()
        return entries.get(token.upper()) or entries.get(token.title()) or entries.get(token)

    @staticmethod
    def _words(text):
        return re.findall(r"[a-z][a-z'-]{1,31}", str(text).lower())

    def semantic_relations(self, token, limit=8):
        """Return a bounded word/synonym/antonym bundle for one entry."""
        entry = self.entry(token)
        if not isinstance(entry, dict):
            return None

        synonyms = []
        antonyms = []
        for value in entry.get("SYNONYMS", []) or []:
            synonyms.extend(self._words(value))
        for value in entry.get("ANTONYMS", []) or []:
            antonyms.extend(self._words(value))
        for meaning in entry.get("MEANINGS", []) or []:
            if not isinstance(meaning, (list, tuple)):
                continue
            if len(meaning) > 2 and isinstance(meaning[2], (list, tuple)):
                for category in meaning[2]:
                    synonyms.extend(self._words(category))

        def unique(items):
            seen = set()
            result = []
            for item in items:
                if item != token and item not in seen:
                    result.append(item)
                    seen.add(item)
                if len(result) >= limit:
                    break
            return result

        return {
            "word": token.lower(),
            "synonyms": unique(synonyms),
            "antonyms": unique(antonyms),
        }
