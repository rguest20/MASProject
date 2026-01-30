import random

STAB_DEFAULTS = {
    "stab_enable": True,

    # how many top tokens we treat as potential monopolies
    "stab_top_n": 5,

    # if top token freq > ratio * median => "monopoly" alert
    "stab_freq_ratio": 3.0,

    # how hard to dampen its links when in monopoly region
    "stab_link_dampen": 0.6,      # 0.6 = 40% reduction

    # gaussian noise on vectors for top-N tokens each tick
    "stab_noise_sigma": 0.01,

    # strength of probability penalty for frequent tokens
    "stab_bias_strength": 2.5,

    # how many semantic ticks a token stays in cooldown
    "stab_cooldown_ticks": 5,

    # extra penalty factor for tokens currently in cooldown
    "stab_cooldown_penalty": 0.4,
}

class SemanticStabilisationMixin:
    """
    Mixin adding semantic stabilisation features to an agent.
    Features:
    - Frequency guard: detects overused tokens and dampens their
      link weights temporarily to prevent monopolies.
    - Phonological noise injection: adds small random noise to
        vectors of frequent tokens to help them escape local minima.
    - Sampling biasing: reduces sampling probabilities of overused
        tokens to encourage diversity.
    Usage:
    - Call _init_semantic_stabilisation() once during agent init.
    - Call semantic_stabilisation_tick() once per semantic tick.
    - Call stab_note_token_usage(token) whenever a token is observed.
    - Call stab_adjust_token_probs(tokens, probs) before sampling
      tokens to adjust their probabilities.
    - Call stab_adjust_link_weight(a, b, w) before applying link
        weights to adjust them.
    """

    # ------------------------------------------------------------
    #  INITIALISATION
    # ------------------------------------------------------------

    def _init_semantic_stabilisation(self):
        """
        Call this once from your semantic system init.
        Ensures we have storage and config for stabilisation.
        """
        # config defaults
        cfg = getattr(self, "config", {})
        for k, v in STAB_DEFAULTS.items():
            cfg.setdefault(k, v)
        self.config = cfg

        sem = self.semantic
        sem.setdefault("_stab_usage_freq", {})   # token -> int
        sem.setdefault("_stab_cooldown", {})     # token -> ticks remaining


    # ------------------------------------------------------------
    #  PUBLIC HOOKS
    # ------------------------------------------------------------

    def semantic_stabilisation_tick(self):
        """
        Call this once per semantic tick, ideally at the end of
        semantic_update_tick().
        """
        if not self.config.get("stab_enable", True):
            return

        self._stab_decay_cooldowns()
        self._stab_frequency_guard()
        self._stab_phonological_noise_injection()


    def stab_note_token_usage(self, token: str):
        """
        Call this from your token observation path, e.g.
        inside _observe_tokens() or similar.
        """
        freq = self.semantic["_stab_usage_freq"]
        freq[token] = freq.get(token, 0) + 1


    def stab_adjust_token_probs(self, tokens, probs):
        """
        Call this from your token sampling path, e.g.
        just before sampling in _sample_token_from_pool().
        Returns a new probs list; tokens list unchanged.
        """
        if (not self.config.get("stab_enable", True)
                or not tokens or not probs):
            return probs

        freq = self.semantic.get("_stab_usage_freq", {})
        if not freq:
            return probs

        max_f = max(freq.values()) if freq else 1
        if max_f <= 0:
            return probs

        bias_strength = self.config["stab_bias_strength"]
        cooldown = self.semantic.get("_stab_cooldown", {})
        cooldown_penalty = self.config["stab_cooldown_penalty"]

        adjusted = []
        for tok, p in zip(tokens, probs):
            f = freq.get(tok, 0)

            # base penalty from frequency
            penalty = 1.0 / (1.0 + (f / max_f) * bias_strength)

            # additional penalty if token is in cooldown
            if tok in cooldown and cooldown[tok] > 0:
                penalty *= cooldown_penalty

            adjusted.append(p * penalty)

        total = sum(adjusted)
        if total <= 0:
            # fall back if something went weird
            return probs

        return [x / total for x in adjusted]


    def stab_adjust_link_weight(self, a, b, w):
        """
        Call this from inside your _link() function before applying w.
        Example:
            w = self.stab_adjust_link_weight(a, b, w)
        """
        if not self.config.get("stab_enable", True):
            return w

        cooldown = self.semantic.get("_stab_cooldown", {})
        if not cooldown:
            return w

        factor = 1.0
        if cooldown.get(a, 0) > 0:
            factor *= self.config["stab_cooldown_penalty"]
        if cooldown.get(b, 0) > 0:
            factor *= self.config["stab_cooldown_penalty"]

        return w * factor


    # ------------------------------------------------------------
    #  INTERNALS: FREQUENCY GUARD & COOLDOWN
    # ------------------------------------------------------------

    def _stab_decay_cooldowns(self):
        cd = self.semantic["_stab_cooldown"]
        to_del = []
        for tok, t in cd.items():
            t -= 1
            if t <= 0:
                to_del.append(tok)
            else:
                cd[tok] = t
        for tok in to_del:
            del cd[tok]


    def _stab_frequency_guard(self):
        """
        Detect frequency monopolies and:
        - dampen their link weights
        - place them in cooldown for a few ticks
        """
        freq = self.semantic.get("_stab_usage_freq", {})
        if not freq:
            return

        values = list(freq.values())
        if len(values) < 2:
            return

        values.sort()
        median = values[len(values) // 2]
        if median <= 0:
            return

        top_n = self.config["stab_top_n"]
        ratio = self.config["stab_freq_ratio"]
        link_dampen = self.config["stab_link_dampen"]
        cooldown_ticks = self.config["stab_cooldown_ticks"]

        top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        max_count = top[0][1]

        if max_count < ratio * median:
            # no strong monopoly this tick
            return

        links = self.semantic.get("links", {})
        cooldown = self.semantic["_stab_cooldown"]

        for tok, _ in top:
            # mark cooldown
            cooldown[tok] = max(cooldown.get(tok, 0), cooldown_ticks)

            # dampen links for this token both ways
            if tok not in links:
                continue
            nbrs = links[tok]
            for nb, entry in list(nbrs.items()):
                # --- dampen forward link ---
                if isinstance(entry, dict):
                    entry["w"] *= link_dampen
                    entry["age"] = max(0, entry.get("age", 0))
                    nbrs[nb] = entry
                else:
                    nbrs[nb] = entry * link_dampen

                # --- dampen backward link ---
                back_row = links.get(nb)
                if back_row and tok in back_row:
                    back_entry = back_row[tok]
                    if isinstance(back_entry, dict):
                        back_entry["w"] *= link_dampen
                        back_entry["age"] = max(0, back_entry.get("age", 0))
                        back_row[tok] = back_entry
                    else:
                        back_row[tok] = back_entry * link_dampen


    # ------------------------------------------------------------
    #  INTERNALS: PHONOLOGICAL NOISE
    # ------------------------------------------------------------

    def _stab_phonological_noise_injection(self):
        """
        Add small vector noise to top-N frequent tokens each tick
        so they can't sit forever at a gravity minimum.
        """
        freq = self.semantic.get("_stab_usage_freq", {})
        if not freq:
            return

        top_n = self.config["stab_top_n"]
        sigma = self.config["stab_noise_sigma"]
        vecs = self.semantic.get("vecs", {})
        if not vecs:
            return

        rng = getattr(self, "local_random", None) or getattr(self, "random", None) or random

        top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        for tok, _ in top:
            v = vecs.get(tok)
            if v is None:
                continue
            vecs[tok] = [val + rng.gauss(0.0, sigma) for val in v]