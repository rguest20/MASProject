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
