"""A public, learned record of conventions that agents have validated."""

from collections import Counter, defaultdict


class CommunityLexicon:
    """Promote repeatedly successful private signals into public conventions.

    This is deliberately not a central dictionary that supplies meanings in
    advance.  A token appears here only after agents communicate it correctly
    in a grounded task.  Agents may still keep local alternatives while a
    convention is uncertain or absent.
    """

    def __init__(self, min_successes=2, min_confidence=0.60, replacement_margin=2):
        self.min_successes = min_successes
        self.min_confidence = min_confidence
        self.replacement_margin = replacement_margin
        self._numeric_evidence = defaultdict(Counter)       # digit -> token -> successes
        self._referential_evidence = defaultdict(Counter)   # referent -> token -> successes
        self._action_evidence = defaultdict(Counter)        # action -> token -> successes
        self._base_evidence = defaultdict(Counter)          # "base" -> base -> successes
        self._grammar_evidence = defaultdict(Counter)       # task type -> order -> successes
        self._numeric_promoted = {}
        self._referential_promoted = {}
        self._action_promoted = {}
        self._base_promoted = {}
        self._grammar_promoted = {}

    @staticmethod
    def _clean_token(token):
        return token.strip().lower() if isinstance(token, str) and token.strip() else None

    def _canonical(self, evidence, promoted, key):
        candidates = evidence.get(key)
        if not candidates:
            return None
        token, support = candidates.most_common(1)[0]
        total = sum(candidates.values())
        current = promoted.get(key)

        if current is None:
            if support < self.min_successes or support / total < self.min_confidence:
                return None
            promoted[key] = token
            return token

        if token == current:
            return current

        # Once a public convention exists, a single competing success should
        # not make the entire population reinterpret a familiar word.  A
        # challenger must have both a clear majority and materially stronger
        # cumulative evidence before replacing it.
        current_support = candidates.get(current, 0)
        if (
            support >= current_support + self.replacement_margin
            and support / total >= self.min_confidence
        ):
            promoted[key] = token
            return token
        return current

    def observe_numeric_success(self, digit, token):
        token = self._clean_token(token)
        if not isinstance(digit, int) or digit < 0 or token is None:
            return False
        previous = self.numeric_token(digit)
        self._numeric_evidence[digit][token] += 1
        return self.numeric_token(digit) != previous

    def observe_referential_success(self, referent, token):
        token = self._clean_token(token)
        if not isinstance(referent, str) or not referent or token is None:
            return False
        previous = self.referential_signal(referent)
        self._referential_evidence[referent][token] += 1
        return self.referential_signal(referent) != previous

    def observe_action_success(self, action, token):
        token = self._clean_token(token)
        if not isinstance(action, str) or not action or token is None:
            return False
        previous = self.action_signal(action)
        self._action_evidence[action][token] += 1
        return self.action_signal(action) != previous

    def numeric_token(self, digit):
        return self._canonical(self._numeric_evidence, self._numeric_promoted, digit)

    def referential_signal(self, referent):
        return self._canonical(
            self._referential_evidence,
            self._referential_promoted,
            referent,
        )

    def action_signal(self, action):
        return self._canonical(self._action_evidence, self._action_promoted, action)

    def numeric_support(self, digit):
        return sum(self._numeric_evidence.get(digit, {}).values())

    def numeric_confidence(self, digit):
        evidence = self._numeric_evidence.get(digit, {})
        total = sum(evidence.values())
        return max(evidence.values()) / total if total else 0.0

    def action_support(self, action):
        return sum(self._action_evidence.get(action, {}).values())

    def observe_base_success(self, base):
        if not isinstance(base, int) or base < 2:
            return False
        previous = self.community_base()
        self._base_evidence["base"][base] += 1
        return self.community_base() != previous

    def community_base(self):
        return self._canonical(self._base_evidence, self._base_promoted, "base")

    def observe_grammar_success(self, task_type, order):
        if not isinstance(task_type, str) or not task_type:
            return False
        order = tuple(order or ())
        if not order:
            return False
        previous = self.grammar_order(task_type)
        self._grammar_evidence[task_type][order] += 1
        return self.grammar_order(task_type) != previous

    def grammar_order(self, task_type):
        return self._canonical(self._grammar_evidence, self._grammar_promoted, task_type)

    def grammar_conventions(self):
        return {
            task_type: order
            for task_type in self._grammar_evidence
            if (order := self.grammar_order(task_type)) is not None
        }

    def referent_for_signal(self, token):
        token = self._clean_token(token)
        if token is None:
            return None
        matches = [
            referent for referent in self._referential_evidence
            if self.referential_signal(referent) == token
        ]
        return matches[0] if len(matches) == 1 else None

    def action_for_signal(self, token):
        token = self._clean_token(token)
        if token is None:
            return None
        matches = [
            action for action in self._action_evidence
            if self.action_signal(action) == token
        ]
        return matches[0] if len(matches) == 1 else None

    def numeric_conventions(self):
        return {
            digit: token
            for digit in self._numeric_evidence
            if (token := self.numeric_token(digit)) is not None
        }

    def referential_conventions(self):
        return {
            referent: token
            for referent in self._referential_evidence
            if (token := self.referential_signal(referent)) is not None
        }

    def action_conventions(self):
        return {
            action: token
            for action in self._action_evidence
            if (token := self.action_signal(action)) is not None
        }

    def referential_support(self, referent):
        return sum(self._referential_evidence.get(referent, {}).values())

    def memory_state(self):
        """Return the durable, community-owned part of the lexicon.

        Private agent vocabularies deliberately stay out of this snapshot.
        Keeping the successful evidence, rather than only the winning token,
        lets a future population retain both a convention and the strength of
        the evidence that earned it.
        """
        return {
            "numeric_evidence": {
                str(key): dict(values) for key, values in self._numeric_evidence.items()
            },
            "referential_evidence": {
                str(key): dict(values) for key, values in self._referential_evidence.items()
            },
            "action_evidence": {
                str(key): dict(values) for key, values in self._action_evidence.items()
            },
            "base_evidence": {
                str(key): int(value) for key, value in self._base_evidence["base"].items()
            },
            "grammar_evidence": {
                str(task_type): [
                    {"order": list(order), "support": int(support)}
                    for order, support in evidence.items()
                ]
                for task_type, evidence in self._grammar_evidence.items()
            },
        }

    def restore_memory_state(self, state):
        """Restore validated convention evidence from a JSON-safe snapshot."""
        if not isinstance(state, dict):
            return

        def restore_token_evidence(source, target, key_parser=lambda key: key):
            if not isinstance(source, dict):
                return
            for raw_key, votes in source.items():
                try:
                    key = key_parser(raw_key)
                except (TypeError, ValueError):
                    continue
                if not isinstance(votes, dict):
                    continue
                for token, support in votes.items():
                    clean = self._clean_token(token)
                    if clean is None:
                        continue
                    try:
                        support = max(0, min(int(support), 1_000_000))
                    except (TypeError, ValueError):
                        continue
                    if support:
                        target[key][clean] += support

        restore_token_evidence(
            state.get("numeric_evidence"), self._numeric_evidence, int
        )
        restore_token_evidence(
            state.get("referential_evidence"), self._referential_evidence
        )
        restore_token_evidence(
            state.get("action_evidence"), self._action_evidence
        )

        base_evidence = state.get("base_evidence")
        if not isinstance(base_evidence, dict):
            base_evidence = {}
        for raw_base, support in base_evidence.items():
            try:
                base = int(raw_base)
                support = max(0, min(int(support), 1_000_000))
            except (TypeError, ValueError):
                continue
            if base >= 2 and support:
                self._base_evidence["base"][base] += support

        grammar_evidence = state.get("grammar_evidence")
        if not isinstance(grammar_evidence, dict):
            grammar_evidence = {}
        for task_type, records in grammar_evidence.items():
            task_type = self._clean_token(task_type)
            if task_type is None or not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                order = tuple(
                    token for token in (self._clean_token(token) for token in record.get("order", []))
                    if token is not None
                )
                try:
                    support = max(0, min(int(record.get("support", 0)), 1_000_000))
                except (TypeError, ValueError):
                    continue
                if order and support:
                    self._grammar_evidence[task_type][order] += support

        # Recompute public conventions from the restored evidence.  This
        # preserves the ordinary confidence and replacement safeguards.
        for digit in self._numeric_evidence:
            self.numeric_token(digit)
        for referent in self._referential_evidence:
            self.referential_signal(referent)
        for action in self._action_evidence:
            self.action_signal(action)
        self.community_base()
        for task_type in self._grammar_evidence:
            self.grammar_order(task_type)
