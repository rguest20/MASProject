"""Small, paced access to a local reading corpus for the community.

The reader deliberately exposes only a short adjacent passage at a time.
It is a source of linguistic context, not an oracle and not a source of
world facts: a fairy-tale sentence should not make a fictional event true in
the community conversation graph.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import re


class CommunityReadingRoom:
    """Select bounded, sequential passages from the user-provided corpus."""

    _SENTENCE = re.compile(r"[^.!?\n]+[.!?]+")
    _TOKEN = re.compile(r"[a-z][a-z'-]{0,31}", re.IGNORECASE)
    _FILENAMES = (
        "clean_merged_fairy_tales_without_eos.txt",
        "cleaned_merged_fairy_tales_without_eos.txt",
    )

    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[1])
        self.path = self._find_corpus()
        self._sentences = None
        self._cursor = None
        self.recent_indices = deque(maxlen=16)

    def _find_corpus(self):
        # Python-owned corpora live beside this package.  The workspace-level
        # fallback preserves an existing user corpus while the project moves
        # into its language-specific directory.
        for root in (self.root, self.root.parent):
            for filename in self._FILENAMES:
                candidate = root / filename
                if candidate.is_file():
                    return candidate
        return None

    @classmethod
    def _clean_sentence(cls, sentence):
        sentence = " ".join(str(sentence or "").split())
        words = cls._TOKEN.findall(sentence)
        # The corpus includes headings, fragments and very long literary
        # lines.  A compact sentence is much more useful for local context
        # and keeps the agents' working memory bounded.
        if not 4 <= len(words) <= 30:
            return None
        return sentence

    def _load(self):
        if self._sentences is not None:
            return self._sentences
        if self.path is None:
            self._sentences = []
            return self._sentences
        try:
            text = self.path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            self._sentences = []
            return self._sentences
        self._sentences = [
            cleaned
            for raw in self._SENTENCE.findall(text)
            if (cleaned := self._clean_sentence(raw)) is not None
        ]
        return self._sentences

    @property
    def available(self):
        return bool(self._load())

    def next_passage(self, sentences=2):
        """Return a small run of neighbouring sentences without repetition."""
        corpus = self._load()
        if not corpus:
            return []
        size = max(1, min(int(sentences), 3, len(corpus)))
        if self._cursor is None:
            self._cursor = random.randrange(len(corpus))
        start = self._cursor
        if start in self.recent_indices and len(corpus) > size:
            start = (start + len(self.recent_indices) + 1) % len(corpus)
        indices = [(start + offset) % len(corpus) for offset in range(size)]
        self._cursor = (start + size) % len(corpus)
        self.recent_indices.extend(indices)
        return [corpus[index] for index in indices]
