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
        candidates = [token]
        lowered = token.lower()
        # Dictionary headwords are commonly singular while ordinary human
        # conversation is not.  This intentionally small fallback avoids
        # pretending to be a full lemmatiser while covering apples/bananas.
        if lowered.endswith("ies") and len(lowered) > 4:
            candidates.append(lowered[:-3] + "y")
        elif lowered.endswith("s") and len(lowered) > 3 and not lowered.endswith("ss"):
            candidates.append(lowered[:-1])
        fallback = None
        for candidate in candidates:
            found = (
                entries.get(candidate.upper())
                or entries.get(candidate.title())
                or entries.get(candidate)
            )
            if found is None:
                continue
            if isinstance(found, dict) and found.get("MEANINGS"):
                return found
            fallback = fallback or found
        return fallback

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

    def meaning_evidence(self, token, token_limit=12, bigram_limit=8):
        """Return bounded, structured evidence from an entry's meanings.

        Category phrases are safer than treating a full definition as a
        sentence template: ``apple`` can support ``edible fruit`` without
        asserting that every adjective in its gloss is universally true.
        Definition words are returned separately at lower confidence for
        semantic linking only.
        """
        entry = self.entry(token)
        if not isinstance(entry, dict):
            return None

        category_tokens = []
        category_bigrams = []
        definition_tokens = []
        for meaning in entry.get("MEANINGS", []) or []:
            if not isinstance(meaning, (list, tuple)):
                continue
            if len(meaning) > 1:
                definition_tokens.extend(self._words(meaning[1]))
            categories = meaning[2] if len(meaning) > 2 and isinstance(meaning[2], (list, tuple)) else []
            for category in categories:
                words = self._words(category)
                category_tokens.extend(words)
                category_bigrams.extend(zip(words, words[1:]))

        def unique(items, limit):
            seen = set()
            result = []
            for item in items:
                if item == token.lower() or item in seen:
                    continue
                seen.add(item)
                result.append(item)
                if len(result) >= limit:
                    break
            return result

        unique_bigrams = []
        seen_bigrams = set()
        for pair in category_bigrams:
            if pair in seen_bigrams or token.lower() in pair:
                continue
            seen_bigrams.add(pair)
            unique_bigrams.append(pair)
            if len(unique_bigrams) >= bigram_limit:
                break
        return {
            "word": token.lower(),
            "category_tokens": unique(category_tokens, token_limit),
            "definition_tokens": unique(definition_tokens, token_limit),
            "category_bigrams": unique_bigrams,
        }
