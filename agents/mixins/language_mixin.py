#!/usr/bin/env python3
"""
agents/mixins/language_mixin.py

Thin compatibility wrapper that delegates language behaviour to cognition-layer
systems (LanguageSystem / PragmaticSystem / IdentitySystem / EpistemicSystem).
"""

import random

from agents.agent_constants import SYLLABLES
from agents.cognition.language_system import LanguageSystem


class LanguageMixin:
    # =====================================================
    # INITIALISATION
    # =====================================================
    def _init_language_system(self):
        if not hasattr(self, "language_system"):
            self.language_system = LanguageSystem(owner=self)
        if hasattr(self, "semantic_system"):
            self.semantic_system.ensure_language_fields()
        self.language_system.init(seed_tokens=100)

    # =====================================================
    # Bridge helpers
    # =====================================================
    def _is_identity_like(self, tok):
        if hasattr(self, "identity_system"):
            return self.identity_system.is_identity_like(tok)
        return False

    def _invent_token(self, *args, **kwargs):
        return self.language_system.invent_token(*args, **kwargs)

    def _observe_language_tokens(self, tokens, gain=None):
        if not tokens:
            return None
        if hasattr(self, "_observe_tokens"):
            return self._observe_tokens(tokens, gain=gain)
        if hasattr(self, "semantic_system"):
            return self.semantic_system.observe_tokens(tokens, gain=gain)
        return None

    def _parse_utterance(self, utterance):
        return self.language_system.parse_utterance(utterance)

    # =====================================================
    # EMOTIONAL MODIFIERS (used by LanguageSystem)
    # =====================================================
    def _emotion_mod_len(self):
        S = self.state
        return int(
            (S["curiosity"] - 0.5) * 3
            + (S["loneliness"] - 0.5) * 2
            + (S["frustration"] - 0.5) * -3
            + (S["happiness"] - 0.5) * 2
        )

    def _emotion_mod_punct(self):
        S = self.state
        return (
            (S["happiness"] - 0.5) * 0.25
            + (S["frustration"] - 0.5) * -0.20
            + (S["confidence"] - 0.5) * 0.15
        )

    # =====================================================
    # LANGUAGE API (compat)
    # =====================================================
    def speak_number(self, n):
        if hasattr(self, "numeric_system"):
            return self.numeric_system.speak_number(n)
        return str(n)

    def produce_utterance(self):
        if hasattr(self, "semantic_system"):
            self.semantic_system.ensure_language_fields()

        traits = getattr(self, "traits", {})
        express = float(traits.get("expressiveness", 0.5))

        base_len = random.randint(2, 6)
        length = max(2, min(10, base_len + int(self._emotion_mod_len())))

        punct_prob = max(
            0.01,
            min(0.9, 0.15 + 0.3 * express + float(self._emotion_mod_punct())),
        )

        syll_pool = list(SYLLABLES)
        invented_pool = [
            t for t in getattr(self, "vocab", set())
            if t not in SYLLABLES and not self._is_identity_like(t)
        ]

        sem = getattr(self, "semantic", {}) or {}
        concept_tokens = set((sem.get("concept_tokens", {}) or {}).values())
        all_tokens = (set(syll_pool) | set(invented_pool)) - concept_tokens

        if hasattr(self, "symbol_map"):
            all_tokens |= (set(getattr(self, "symbol_map", {}).values()) - concept_tokens)
        if hasattr(self, "reasoning_tokens"):
            all_tokens |= set(getattr(self, "reasoning_tokens", {}).values())

        all_tokens = {
            t for t in all_tokens
            if isinstance(t, str) and t.strip() and not self._is_identity_like(t)
        }
        all_tokens.add("why")

        if hasattr(self, "semantic_system"):
            for t in all_tokens:
                self.semantic_system._ensure_vec(t)
        all_tokens = list(all_tokens)

        if not all_tokens:
            utter = "na"
            self._last_tokens = ["na"]
            self.last_written_word = utter
            mem = self.utterance_memory["usage_count"]
            mem[utter] = mem.get(utter, 0) + 1
            return utter

        prefs = self.utter_bias["symbol_preferences"]
        for t in all_tokens:
            prefs.setdefault(t, 0.2)
        for w in invented_pool:
            prefs.setdefault(w, 0.8)
        if hasattr(self, "symbol_map"):
            for tok in getattr(self, "symbol_map", {}).values():
                prefs.setdefault(tok, 0.6)
        if hasattr(self, "reasoning_tokens"):
            for r in getattr(self, "reasoning_tokens", {}).values():
                prefs.setdefault(r, 0.05)

        for t in getattr(self, "recent_tokens", [])[-25:]:
            if not self._is_identity_like(t):
                prefs[t] = min(3.0, prefs.get(t, 0.2) + 0.15)
        if random.random() < express:
            for w in invented_pool:
                prefs[w] = min(3.0, prefs.get(w, 1.2) + 0.15)

        toks: list[str] = []
        cur = self.language_system.choose_utter_start(all_tokens)
        if not isinstance(cur, str) or not cur.strip():
            cur = random.choice(all_tokens)
        toks.append(cur)
        if not self._is_identity_like(cur):
            self.recent_tokens.append(cur)
            prefs[cur] = min(3.0, prefs.get(cur, 0.2) + 0.05)

        for _ in range(length - 1):
            neighbors: list[str] = []
            if hasattr(self, "semantic_system"):
                neighbors = self.semantic_system._semantic_neighbors(cur, k=15, max_radius=0.7)
                fam_nbrs = self.semantic_system._family_neighbors(cur)
                if fam_nbrs:
                    neighbors = list(set(neighbors) | set(fam_nbrs))
            if neighbors and random.random() < 0.2:
                extra = random.sample(all_tokens, min(5, len(all_tokens)))
                neighbors = list(set(neighbors) | set(extra))

            candidate_pool = neighbors or all_tokens
            tok = self.language_system.sample_token_from_pool(candidate_pool, prefs)
            if tok is None:
                tok = random.choice(all_tokens)
            toks.append(tok)
            if not self._is_identity_like(tok):
                self.recent_tokens.append(tok)
                prefs[tok] = min(3.0, prefs.get(tok, 0.2) + 0.05)
            cur = tok

        toks = [t for t in toks if isinstance(t, str) and t.strip()] or ["na"]
        utter = self.language_system.render_utterance(toks, punct_prob=punct_prob)

        clean = self._parse_utterance(utter)
        if clean:
            self._observe_language_tokens(clean)
            if len(clean) > 1 and hasattr(self, "symbol_map"):
                symvals = set(getattr(self, "symbol_map", {}).values())
                if symvals and all(tok in symvals for tok in clean):
                    self._observe_language_tokens(clean, gain=0.2)

        self._last_tokens = clean
        self.last_written_word = utter
        mem = self.utterance_memory["usage_count"]
        mem[utter] = mem.get(utter, 0) + 1
        return utter

    def language_world_ingest_step(self):
        if not hasattr(self, "api") or self.api is None:
            return

        if hasattr(self, "semantic_system"):
            self.semantic_system.ensure_language_fields()

        texts = []
        for fname in ["/notes.txt", "/help_responses.txt", "/family_gossip.txt"]:
            t = self.api.read_text(fname)
            if t:
                texts.append(t)
        if not texts:
            return

        numeric_tokens = set(getattr(self, "numeric_semantic", {}).keys())

        for txt in texts:
            for line in txt.strip().splitlines()[-10:]:
                if "teach_numeric" in line and hasattr(self, "_integrate_teach_numeric_line"):
                    try:
                        self._integrate_teach_numeric_line(line)
                    except Exception:
                        pass

                toks: list[str] = []
                for t in line.split():
                    raw = t.strip(",.!?;:\"'")
                    if not raw:
                        continue
                    if self._is_identity_like(raw.lower()):
                        continue

                    tok = raw.lower()
                    if tok in self.vocab or tok in getattr(self, "symbol_map", {}).values():
                        toks.append(tok)

                for t in toks:
                    self.vocab.add(t)
                    if hasattr(self, "semantic_system"):
                        self.semantic_system._ensure_vec(t)

                if toks:
                    toks = [t for t in toks if t not in numeric_tokens]
                    self._observe_language_tokens(
                        toks, gain=getattr(self, "semantic", {}).get("dictionary_gain", 0.1)
                    )

    def get_overall_expectation(self):
        return self.language_system.get_overall_expectation()

    def get_utterance_expectation(self, utterance):
        return self.language_system.get_utterance_expectation(utterance)

    def learn_from_feedback(self, utterance, reward, lr=0.1):
        return self.language_system.learn_from_feedback(utterance, reward, lr=lr)

    def apply_flavour_homeostasis(
        self,
        community_map,
        soft_strength=0.15,
        dominance_thresh=0.22,
        scarcity_thresh=0.05,
        min_usage=3,
    ):
        if hasattr(self, "epistemic_system"):
            return self.epistemic_system.apply_flavour_homeostasis(
                community_map,
                soft_strength=soft_strength,
                dominance_thresh=dominance_thresh,
                scarcity_thresh=scarcity_thresh,
                min_usage=min_usage,
            )
        return None

    # =====================================================
    # SOCIAL DIALOGUE BEHAVIOUR (compat)
    # =====================================================
    def _get_trust_to(self, other_id):
        if hasattr(self, "pragmatic_system"):
            return self.pragmatic_system.get_trust_to(other_id)
        return 0.0

    def _base_talk_probability(self):
        if hasattr(self, "pragmatic_system"):
            return self.pragmatic_system.base_talk_probability()
        return 0.0

    def _social_affinity(self, other):
        if hasattr(self, "pragmatic_system"):
            return self.pragmatic_system.social_affinity(other)
        return 0.0

    def choose_conversation_partner(self, agents):
        if hasattr(self, "pragmatic_system"):
            return self.pragmatic_system.choose_conversation_partner(agents)
        return None

    def receive_message(self, from_id, utterance):
        if hasattr(self, "pragmatic_system"):
            return self.pragmatic_system.receive_message(from_id, utterance)
        return None

    def mark_spoken_turn(self):
        if hasattr(self, "pragmatic_system"):
            return self.pragmatic_system.mark_spoken_turn()
        return None

    def get_or_create_nickname_for_id(self, other_id: int) -> str:
        if hasattr(self, "identity_system"):
            return self.identity_system.get_or_create_nickname_for_id(other_id)
        return f"agent_{other_id}"

    def get_or_create_nickname_for_agent(self, other):
        if hasattr(self, "identity_system"):
            return self.identity_system.get_or_create_nickname_for_agent(other)
        return None

    def _observe_nickname_use(self, from_id: int, utterance: str):
        if hasattr(self, "identity_system"):
            return self.identity_system.observe_nickname_use(from_id, utterance)
        return None

    def produce_addressed_utterance(self, listener) -> str:
        if hasattr(self, "pragmatic_system"):
            return self.pragmatic_system.produce_addressed_utterance(listener)
        return self.produce_utterance() or ""


LanguageMixinV2 = LanguageMixin
