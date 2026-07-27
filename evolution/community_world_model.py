"""Small, shared concept graph for the human conversation bridge.

The graph is deliberately conservative: dictionary data creates hypotheses,
human statements create tentative assertions, and explicit feedback determines
which relations become useful in later replies.  It is not an English
knowledge base or a source of unrestricted synonym substitutions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re


class CommunityWorldModel:
    _STATEMENT = re.compile(
        r"(?:^|[.!?]\s*)(?:(?:the|an|a)\s+)?([a-z][a-z'-]{1,31})\s+"
        r"(is|are|has|have)\s+(?:(?:the|an|a)\s+)?([a-z][a-z'-]{1,31})",
        re.IGNORECASE,
    )

    def __init__(self):
        self.edges = Counter()  # (relation, subject, object) -> confidence
        self.sources = defaultdict(Counter)
        self._load_primer()

    def _load_primer(self):
        """Load a small editable set of tentative world facts, if present."""
        path = Path(__file__).resolve().parents[1] / "world_primer.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for fact in payload.get("facts", []) if isinstance(payload, dict) else []:
            if not isinstance(fact, dict):
                continue
            self._adjust(
                fact.get("relation", "related_to"),
                fact.get("subject"),
                fact.get("object"),
                float(fact.get("confidence", 0.0)),
                "primer",
            )

    @staticmethod
    def _normalise(word):
        return str(word or "").lower().strip()

    def _adjust(self, relation, subject, obj, delta, source):
        subject = self._normalise(subject)
        obj = self._normalise(obj)
        if not subject or not obj or subject == obj:
            return
        key = (relation, subject, obj)
        value = max(-1.0, min(2.0, float(self.edges[key]) + float(delta)))
        if value <= 0.01:
            self.edges.pop(key, None)
            self.sources.pop(key, None)
            return
        self.edges[key] = value
        self.sources[key][source] += abs(float(delta))

    def observe_dictionary(self, word, meaning=None, relations=None):
        """Add cautious category and opposition hypotheses from a lookup."""
        word = self._normalise(word)
        if not word:
            return
        for category in (meaning or {}).get("category_tokens", []):
            # Category headings are the least ambiguous structured field in
            # filtered.json.  They are hypotheses until corroborated.
            self._adjust("is_a", word, category, +0.12, "dictionary_category")
        for opposite in (relations or {}).get("antonyms", []):
            self._adjust("opposite", word, opposite, +0.10, "dictionary_antonym")
            self._adjust("opposite", opposite, word, +0.10, "dictionary_antonym")
        # Raw synonyms stay in each agent's semantic evidence.  They are too
        # sense-ambiguous to become shared world facts (``game -> plot`` is a
        # valid dictionary sense but not evidence about this conversation).

    def observe_human_text(self, text):
        """Record bounded copular/possessive assertions as tentative facts."""
        text = str(text or "").lower()
        # A question such as ``what is banana?`` has the same surface shape
        # as an assertion.  Treating it as one would make ``what`` a real
        # subject in the graph and later contaminate answers.
        if "?" in text:
            return
        for match in self._STATEMENT.finditer(text):
            subject, verb, obj = match.groups()
            if subject in {"what", "who", "where", "when", "why", "how"}:
                continue
            relation = "has" if verb in {"has", "have"} else "has_property"
            # If the dictionary already presents the target as a category for
            # the subject, this is better modelled as a class relation.
            if any(
                rel == "is_a" and left == subject and right == obj
                for rel, left, right in self.edges
            ):
                relation = "is_a"
            self._adjust(relation, subject, obj, +0.18, "human_assertion")

    def apply_feedback(self, text, reward):
        """Confirm or reject any simple relation expressed by a reply."""
        reward = 1.0 if reward > 0 else -1.0
        for match in self._STATEMENT.finditer(str(text or "").lower()):
            subject, verb, obj = match.groups()
            relation = "has" if verb in {"has", "have"} else "has_property"
            matches = [
                key for key in self.edges
                if key[1] == subject and key[2] == obj
            ]
            if matches:
                for key in matches:
                    self._adjust(key[0], subject, obj, 0.40 * reward, "human_feedback")
            else:
                self._adjust(relation, subject, obj, 0.40 * reward, "human_feedback")

    def candidates(self, subject, minimum=0.18):
        subject = self._normalise(subject)
        result = {}
        for (relation, left, right), confidence in self.edges.items():
            if left != subject or confidence < minimum:
                continue
            if relation not in {"is_a", "has_property", "has", "opposite"}:
                continue
            result[right] = max(result.get(right, 0.0), float(confidence))
        return result

    def facts(self, subject, relation=None, minimum=0.50):
        """Return supported facts for one subject, strongest first.

        This is deliberately subject-first: a fact about ``banana`` must not
        become an available completion merely because the current sentence
        happens to contain the word ``is``.
        """
        subject = self._normalise(subject)
        if relation in {"is", "are"}:
            relations = {"is_a", "has_property", "opposite"}
        elif relation in {"has", "have"}:
            relations = {"has"}
        else:
            relations = {"is_a", "has_property", "has", "opposite"}
        result = [
            (edge_relation, right, float(confidence))
            for (edge_relation, left, right), confidence in self.edges.items()
            if left == subject and edge_relation in relations and confidence >= minimum
        ]
        return sorted(result, key=lambda item: (-item[2], item[0], item[1]))

    def subjects(self, relation, obj, minimum=0.50):
        """Return subjects with a supported relation to ``obj``."""
        obj = self._normalise(obj)
        if relation in {"is", "are"}:
            relations = {"is_a", "has_property", "opposite"}
        elif relation in {"has", "have"}:
            relations = {"has"}
        else:
            relations = {"is_a", "has_property", "has", "opposite"}
        result = [
            (left, edge_relation, float(confidence))
            for (edge_relation, left, right), confidence in self.edges.items()
            if right == obj and edge_relation in relations and confidence >= minimum
        ]
        return sorted(result, key=lambda item: (-item[2], item[0], item[1]))

    @staticmethod
    def render_fact(subject, relation, obj):
        """Render one compact surface form for a learned graph triple."""
        verb = "has" if relation == "has" else "is"
        if relation == "opposite":
            return f"{subject} is opposite {obj}."
        return f"{subject} {verb} {obj}."

    def supports_clause(self, subject, verb, obj, minimum=0.18):
        """Whether a simple surface clause is backed by this exact subject."""
        subject = self._normalise(subject)
        obj = self._normalise(obj)
        relations = {"has"} if verb in {"has", "have"} else {"is_a", "has_property", "opposite"}
        return any(
            self.edges.get((relation, subject, obj), 0.0) >= minimum
            for relation in relations
        )

    def continuations(self, word, allowed_words, subject=None, minimum=0.18):
        """Return graph-supported continuations within one active subject.

        When a subject is active, the graph supplies a relation word after
        that subject and an object only after that relation word.  The old
        behaviour treated every ``is`` in the graph as equivalent, allowing
        ``apple is`` to continue with facts belonging to banana or game.
        """
        word = self._normalise(word)
        subject = self._normalise(subject)
        result = []
        if subject:
            facts = self.facts(subject, minimum=minimum)
            if word == subject:
                if any(relation == "has" for relation, _, _ in facts):
                    result.append("has")
                if any(relation != "has" for relation, _, _ in facts):
                    result.append("is")
            elif word in {"is", "are", "has", "have"}:
                relation = "has" if word in {"has", "have"} else "is"
                result.extend(obj for _, obj, _ in self.facts(
                    subject, relation=relation, minimum=minimum
                ))
        else:
            result = list(self.candidates(word, minimum))
        return list(dict.fromkeys(item for item in result if item in allowed_words))

    def relation_score(self, words):
        score = 0.0
        for left, right in zip(words, words[1:]):
            for relation in ("is_a", "has_property", "has", "opposite"):
                score += float(self.edges.get((relation, left, right), 0.0))
        for subject, copula, obj in zip(words, words[1:], words[2:]):
            if copula not in {"is", "are", "has", "have"}:
                continue
            relation = "has" if copula in {"has", "have"} else "has_property"
            score += float(self.edges.get((relation, subject, obj), 0.0))
            score += float(self.edges.get(("is_a", subject, obj), 0.0))
        return score / max(1, len(words) - 1)

    def metrics(self):
        confirmed = sum(value >= 0.50 for value in self.edges.values())
        relations = Counter(relation for relation, _, _ in self.edges)
        return {
            "edges": len(self.edges),
            "confirmed": confirmed,
            "is_a": int(relations.get("is_a", 0)),
            "properties": int(relations.get("has_property", 0)),
            "opposites": int(relations.get("opposite", 0)),
        }
