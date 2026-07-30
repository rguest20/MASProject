import random

from evolution.coordinator_settings import FOCUSED_GROUNDING_EXPERIMENT


class CoordinatorTaskSymbolGenerationMixin:
    def generate_numeric_compare_task(self):
        if not self.agents:
            return None

        ref = random.choice(self.agents)
        smap = getattr(ref, "symbol_map", None)
        if not smap:
            return None

        base = getattr(ref.counting, "base", 16)
        max_n = base ** 2 + random.randint(0, base * 4)

        A_val = random.randint(0, max_n)
        B_val = random.randint(0, max_n)

        A_tokens = ref.numeric_system.speak_number(A_val).split()
        B_tokens = ref.numeric_system.speak_number(B_val).split()

        A_str = " ".join(A_tokens)
        B_str = " ".join(B_tokens)

        if A_val > B_val:
            correct_phrase = A_str
        elif B_val > A_val:
            correct_phrase = B_str
        else:
            correct_phrase = A_str

        return {
            "task_id": self._generate_task_id(),
            "task_type": "compare_numbers",
            "instruction": {
                "base": base,
                "symbol_map": smap,
                "format": "compare",
                "description": "Which number is larger?"
            },
            "data": {"A": A_str, "B": B_str},
            "ground_truth": {
                "ref_agent": ref.id,
                "A_val": A_val,
                "B_val": B_val,
                "answer_phrase": correct_phrase
            },
            "responses": [],
        }

    def generate_agreement_dialogue_task(self):
        if len(self.agents) < 2:
            return None

        a, b = random.sample(self.agents, 2)
        tid = self._generate_task_id()

        try:
            u1 = a.produce_utterance()
            u2 = b.produce_utterance()
        except Exception:
            u1, u2 = "su", "tol"

        return {
            "task_id": tid,
            "task_type": "agreement_dialogue",
            "topic": f"{u1} || {u2}",
            "assigned_agents": [a.id, b.id],
        }

    def generate_reconcile_counts_task(self, force_fresh=False):
        if len(self.agents) < 2:
            return None

        by_id = {agent.id: agent for agent in self.agents}
        ledger = getattr(self, "community_lexicon", None)
        public_base = ledger.community_base() if ledger is not None else None
        memories = [
            item for item in getattr(self, "numeric_memory", [])
            if item["a_id"] in by_id
            and item["b_id"] in by_id
            and item["value"] < (
                public_base or min(by_id[item["a_id"]].counting.base, by_id[item["b_id"]].counting.base)
            )
        ]

        def curriculum_weight(value):
            support = ledger.numeric_support(value) if ledger is not None else 0
            promoted = ledger.numeric_token(value) if ledger is not None else None
            # Unpromoted digits get most of the curriculum.  Rehearsal remains
            # possible, but strong conventions no longer consume every trial.
            return 8.0 / (1.0 + support) if promoted is None else max(0.25, 2.0 / (1.0 + support))

        # Rehearsal is essential here: a correction needs a later test of the
        # same peer signal before it can influence selection.  New pairings
        # still provide the path by which a convention spreads outward.
        if memories and not force_fresh and random.random() < 0.65:
            recent_memories = memories[-120:]
            item = random.choices(
                recent_memories,
                weights=[curriculum_weight(memory["value"]) for memory in recent_memories],
                k=1,
            )[0]
            a = by_id[item["a_id"]]
            b = by_id[item["b_id"]]
            val = item["value"]
            A_view = item["a_signal"]
            B_view = item["b_signal"]
        else:
            a, b = random.sample(self.agents, 2)

            # Keep this a single digit so feedback can unambiguously ground a
            # peer's self-invented numeral in the listener's map.
            shared_base = public_base or min(a.counting.base, b.counting.base)
            candidates = list(range(max(1, shared_base)))
            val = random.choices(
                candidates,
                weights=[curriculum_weight(value) for value in candidates],
                k=1,
            )[0]

            def encode(agent, value):
                try:
                    toks = agent.numeric_system.speak_number(value).split()
                except Exception:
                    toks = [agent.numeric_system.get_symbol(value)]
                return " ".join(toks)

            A_view = encode(a, val)
            B_view = encode(b, val)
            self.numeric_memory.append({
                "a_id": a.id,
                "b_id": b.id,
                "value": val,
                "a_signal": A_view,
                "b_signal": B_view,
            })
            self.numeric_memory = self.numeric_memory[-240:]

        tid = self._generate_task_id()

        return {
            "task_id": tid,
            "task_type": "reconcile_counts",
            "value": val,
            # Each agent receives the other agent's signal rather than its
            # own encoding, creating genuine translation pressure.
            "views": {a.id: B_view, b.id: A_view},
            "signal_sources": {a.id: b.id, b.id: a.id},
            "assigned_agents": [a.id, b.id],
            "responses": []
        }

    def generate_translate_number_task(self):
        # Direct, fresh single-symbol checks remain in the curriculum after a
        # base is learned.  Otherwise an early near-complete map can hide one
        # weak or missing numeral forever behind multi-digit translations.
        task = self.generate_reconcile_counts_task(force_fresh=True)
        if task:
            task["task_type"] = "translate_number"
        return task

    def generate_translate_quantity_task(self):
        """Translate a multi-digit quantity, allowing a public base to emerge."""
        if len(self.agents) < 2:
            return None

        ledger = getattr(self, "community_lexicon", None)
        shared_base = ledger.community_base() if ledger is not None else None
        if shared_base is not None:
            a, b = random.sample(self.agents, 2)
            base = shared_base
        else:
            groups = {}
            for agent in self.agents:
                groups.setdefault(agent.counting.base, []).append(agent)
            viable = [(base, group) for base, group in groups.items() if len(group) >= 2]
            if not viable:
                return None
            base, group = random.choice(viable)
            a, b = random.sample(group, 2)

        value = random.randint(base, base * base - 1)

        def encode(agent):
            return agent.numeric_system.speak_number(value, base=base)

        a_signal = encode(a)
        b_signal = encode(b)
        return {
            "task_id": self._generate_task_id(),
            "task_type": "translate_quantity",
            "value": value,
            "numeric_base": base,
            "views": {a.id: b_signal, b.id: a_signal},
            "signal_sources": {a.id: b.id, b.id: a.id},
            "assigned_agents": [a.id, b.id],
            "responses": [],
        }

    def generate_compositional_signal_task(self):
        """Ground a short referent-plus-quantity message and its word order."""
        ledger = getattr(self, "community_lexicon", None)
        if len(self.agents) < 2 or ledger is None:
            return None
        numeric = ledger.numeric_conventions()
        referential = ledger.referential_conventions()
        if not numeric or not referential:
            return None

        speaker, listener = random.sample(self.agents, 2)
        referent, referent_token = random.choice(list(referential.items()))
        value, number_token = random.choice(list(numeric.items()))
        order = ledger.grammar_order("referent_quantity") or random.choice([
            ("referent", "number"),
            ("number", "referent"),
        ])
        values = {"referent": referent_token, "number": number_token}
        signal = " ".join(values[role] for role in order)
        return {
            "task_id": self._generate_task_id(),
            "task_type": "compositional_signal",
            "assigned_agents": [listener.id],
            "data": {
                "speaker_id": speaker.id,
                "referent": referent,
                "value": value,
                "signal": signal,
                "order": order,
            },
            "responses": [],
        }

    def generate_referential_signal_task(self):
        """Ask one agent to interpret another agent's invented word."""
        if len(self.agents) < 2:
            return None

        by_id = {agent.id: agent for agent in self.agents}
        memories = [
            item for item in getattr(self, "referential_memory", [])
            if item["speaker_id"] in by_id and item["listener_id"] in by_id
        ]

        def curriculum_weight(referent):
            ledger = getattr(self, "community_lexicon", None)
            support = ledger.referential_support(referent) if ledger is not None else 0
            promoted = ledger.referential_signal(referent) if ledger is not None else None
            return 8.0 / (1.0 + support) if promoted is None else max(0.25, 2.0 / (1.0 + support))

        # Most trials rehearse a recent signal.  This gives a learned mapping
        # a fair chance to affect fitness instead of relying on a rare random
        # re-encounter of the exact same speaker/listener/referent triple.
        if memories and random.random() < 0.65:
            recent_memories = memories[-60:]
            item = random.choices(
                recent_memories,
                weights=[curriculum_weight(memory["referent"]) for memory in recent_memories],
                k=1,
            )[0]
            speaker = by_id[item["speaker_id"]]
            listener = by_id[item["listener_id"]]
            referent = item["referent"]
            signal = item["signal"]
        else:
            speaker, listener = random.sample(self.agents, 2)
            referent = random.choices(
                [f"r{index}" for index in range(6)],
                weights=[curriculum_weight(f"r{index}") for index in range(6)],
                k=1,
            )[0]
            lexicon = speaker.referent_lexicon
            ledger = getattr(self, "community_lexicon", None)
            signal = ledger.referential_signal(referent) if ledger is not None else None
            signal = signal or lexicon.get(referent)
            if not signal:
                signal = speaker._invent_token(max_syllables=2)
                lexicon[referent] = signal
                speaker.vocab.add(signal)
                speaker.semantic_system.ensure_vec(signal)
                speaker._ensure_token_semantic(signal)

            self.referential_memory.append({
                "speaker_id": speaker.id,
                "listener_id": listener.id,
                "referent": referent,
                "signal": signal,
            })
            self.referential_memory = self.referential_memory[-120:]

        return {
            "task_id": self._generate_task_id(),
            "task_type": "referential_signal",
            "assigned_agents": [listener.id],
            "data": {
                "speaker_id": speaker.id,
                "referent": referent,
                "signal": signal,
            },
            "responses": [],
        }

    def generate_action_signal_task(self, force_fresh=False):
        """Ground an opaque action signal before using it in longer messages."""
        if len(self.agents) < 2:
            return None

        by_id = {agent.id: agent for agent in self.agents}
        memories = [
            item for item in getattr(self, "action_memory", [])
            if item["speaker_id"] in by_id and item["listener_id"] in by_id
        ]
        ledger = getattr(self, "community_lexicon", None)

        if memories and not force_fresh and random.random() < 0.65:
            item = random.choice(memories[-60:])
            speaker = by_id[item["speaker_id"]]
            listener = by_id[item["listener_id"]]
            action = item["action"]
            signal = item["signal"]
        else:
            speaker, listener = random.sample(self.agents, 2)
            def action_weight(action):
                support = ledger.action_support(action) if ledger is not None else 0
                promoted = ledger.action_signal(action) if ledger is not None else None
                return 8.0 / (1.0 + support) if promoted is None else max(0.25, 2.0 / (1.0 + support))

            action = random.choices(
                [f"a{index}" for index in range(4)],
                weights=[action_weight(f"a{index}") for index in range(4)],
                k=1,
            )[0]
            signal = ledger.action_signal(action) if ledger is not None else None
            signal = signal or speaker.action_lexicon.get(action)
            if not signal:
                signal = speaker._invent_token(max_syllables=2)
                speaker.action_lexicon[action] = signal
                speaker.vocab.add(signal)
                speaker.semantic_system.ensure_vec(signal)
                speaker._ensure_token_semantic(signal)
            self.action_memory.append({
                "speaker_id": speaker.id,
                "listener_id": listener.id,
                "action": action,
                "signal": signal,
            })
            self.action_memory = self.action_memory[-120:]

        return {
            "task_id": self._generate_task_id(),
            "task_type": "action_signal",
            "assigned_agents": [listener.id],
            "data": {"speaker_id": speaker.id, "action": action, "signal": signal},
            "responses": [],
        }

    def generate_action_maintenance_task(self):
        """Keep expanding action vocabulary after the first two conventions."""
        return self.generate_action_signal_task(force_fresh=True)

    def generate_compositional_action_signal_task(self):
        """Combine public referent, action, and number conventions in one message."""
        ledger = getattr(self, "community_lexicon", None)
        if len(self.agents) < 2 or ledger is None:
            return None
        numeric = ledger.numeric_conventions()
        referential = ledger.referential_conventions()
        actions = ledger.action_conventions()
        if not numeric or not referential or not actions:
            return None

        speaker, listener = random.sample(self.agents, 2)
        referent, referent_token = random.choice(list(referential.items()))
        action, action_token = random.choice(list(actions.items()))
        value, number_token = random.choice(list(numeric.items()))
        order = ledger.grammar_order("referent_action_number")
        if order is None:
            # Extend the already learned two-slot construction rather than
            # restarting grammar search from a uniformly random permutation.
            two_slot_order = ledger.grammar_order("referent_quantity")
            if two_slot_order == ("referent", "number"):
                order = ("referent", "action", "number")
            elif two_slot_order == ("number", "referent"):
                order = ("number", "referent", "action")
            else:
                order = random.choice([
                    ("referent", "action", "number"),
                    ("number", "referent", "action"),
                    ("action", "referent", "number"),
                ])
        signals = {"referent": referent_token, "action": action_token, "number": number_token}
        return {
            "task_id": self._generate_task_id(),
            "task_type": "compositional_action_signal",
            "assigned_agents": [listener.id],
            "data": {
                "speaker_id": speaker.id,
                "referent": referent,
                "action": action,
                "value": value,
                "signal": " ".join(signals[role] for role in order),
                "order": order,
            },
            "responses": [],
        }
