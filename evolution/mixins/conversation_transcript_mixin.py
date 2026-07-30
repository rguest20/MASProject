"""A small, safe bridge between a human transcript and community language.

The bridge keeps structured simulation commands separate, but ordinary human
sentences are clustered into community-created, opaque interaction modes.  It
never treats unrecognised prose as an instruction or overwrites human text.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
from pathlib import Path
import random
import re

from evolution.community_world_model import CommunityWorldModel


class ConversationTranscriptMixin:
    @staticmethod
    def _signature(line_number, text):
        raw = f"{line_number}:{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _transcript(self):
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            return None

    def _latest_unanswered_prompt(self, text):
        """Return the most recent complete Ryan line without a reply."""
        entries = []
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            match = self._PREFIX.match(raw_line)
            if not match:
                continue
            role, body = match.groups()
            body = body.strip()
            if body:
                entries.append((line_number, role.lower(), body))

        latest = None
        for index, (line_number, role, body) in enumerate(entries):
            if role != "ryan":
                continue
            next_role = entries[index + 1][1] if index + 1 < len(entries) else None
            if next_role != "community":
                latest = (line_number, body)
        return latest

    def _aliases_from_transcript(self, text):
        aliases = {}
        for raw_line in text.splitlines():
            match = self._PREFIX.match(raw_line)
            if not match or match.group(1).lower() != "ryan":
                continue
            body = match.group(2).strip()
            entity = self._TEACH_ENTITY.match(body)
            if entity:
                key, word = entity.groups()
                aliases[word.lower()] = key.lower()
                continue
            number = self._TEACH_NUMBER.match(body)
            if number:
                value, word = number.groups()
                aliases[word.lower()] = int(value)
        return aliases

    @staticmethod
    def _agreement(values, population_size):
        values = [value for value in values if value]
        if not values:
            return None, 0, population_size
        choice, count = Counter(values).most_common(1)[0]
        return choice, count, population_size

    @staticmethod
    def _resolve(value, aliases, prefix=None):
        value = (value or "").strip().lower()
        resolved = aliases.get(value, value)
        if prefix and isinstance(resolved, str) and not resolved.startswith(prefix):
            return None
        return resolved

    @staticmethod
    def _resolve_number(value, aliases):
        value = (value or "").strip().lower()
        resolved = aliases.get(value, value)
        if isinstance(resolved, int):
            return resolved
        try:
            return int(resolved)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _answer_text(body):
        return f"Community: {body}\nRyan: \n"

    @staticmethod
    def _human_tokens(prompt):
        """Keep only bounded word-like tokens; transcript text is never executed."""
        return re.findall(r"[a-z][a-z'-]{0,31}", prompt.lower())[:12]

    def _world_question_reply(self, coordinator, prompt):
        """Answer a simple graph question from one subject-bound fact set.

        A question is still observed by the community's normal intent and
        working-memory machinery.  Only the final retrieval is constrained:
        the answer can use a fact about the requested subject (or, for a
        possessive reverse query, a subject that owns the requested object).
        """
        cleaned = prompt.strip()
        reverse = self._WORLD_REVERSE_QUERY.match(cleaned)
        direct = self._WORLD_SUBJECT_QUERY.match(cleaned)
        if reverse is None and direct is None:
            return None

        tokens = self._human_tokens(cleaned)
        self._update_conversation_state(coordinator, cleaned, tokens)
        self._ingest_human_tokens(coordinator, tokens, prompt=cleaned)

        if reverse is not None:
            verb, obj = reverse.groups()
            subjects = self.world_model.subjects(verb, obj, minimum=0.50)
            if subjects:
                subject, relation, _ = subjects[0]
                answer = self.world_model.render_fact(subject, relation, obj)
            else:
                answer = f"I do not yet have a confirmed fact involving {obj.lower()}."
        else:
            verb, subject = direct.groups()
            facts = self.world_model.facts(subject, relation=verb, minimum=0.50)
            if facts:
                relation, obj, _ = facts[0]
                answer = self.world_model.render_fact(subject.lower(), relation, obj)
            else:
                answer = f"I do not yet have a confirmed fact about {subject.lower()}."

        self._record_community_response(
            answer,
            mode=self.active_intent_mode,
            topic=self.active_topic,
        )
        return answer, 1.0
