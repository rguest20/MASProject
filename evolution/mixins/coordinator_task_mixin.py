import random

from evolution.coordinator_settings import FOCUSED_GROUNDING_EXPERIMENT


class CoordinatorTaskMixin:
    """
    Task generation, sampling, and scoring helpers extracted from the
    monolithic Coordinator.  Keeping them in a mixin lets the core
    class focus on orchestration while reuse remains straightforward.
    """

    def _generate_task_id(self):
        tid = f"T{self.next_task_id:04d}"
        self.next_task_id += 1
        return tid

    # ---------------------------------------------------------
    #  Archetype stats / homeostasis
    # ---------------------------------------------------------
    def _update_archetype_stats(self):
        """
        Compute current distribution of archetypes over the population.
        Expects each agent to expose `emergent_archetype`.
        """
        counts = {
            "explorer": 0,
            "cooperator": 0,
            "habit": 0,
            "loner": 0,
        }

        for ag in self.agents:
            label = getattr(ag, "emergent_archetype", None)
            if label is None:
                label = getattr(ag, "archetype_label", None)

            if label in counts:
                counts[label] += 1

        total = sum(counts.values())
        if total == 0:
            return

        self.archetype_stats = {k: counts[k] / total for k in counts}

    def classify_agent_archetype(self, ag):
        nov = float(ag.traits.get("novelty_weight", 0.5))
        coop = float(ag.traits.get("cooperation_weight", 0.5))
        stab = float(ag.traits.get("stability_weight", 0.5))
        trust = float(ag.traits.get("trust_threshold", 0.5))

        if nov > 0.60 and stab < 0.40:
            return "explorer"

        if coop > 0.60 and trust < 0.65:
            return "cooperator"

        if stab > 0.60:
            return "habit"

        if trust > 0.70 and coop < 0.40:
            return "loner"

        return "cooperator"

    def assign_archetypes(self, population, classifier_fn):
        for ag in population:
            ag.emergent_archetype = classifier_fn(ag)

    # ---------------------------------------------------------
    #  Homeostatic task weighting / sampling
    # ---------------------------------------------------------
    def _update_task_weights_homeostasis(self, verbose=False):
        self._update_archetype_stats()
        dist = self.archetype_stats

        cfg = self.homeostasis_cfg
        low = cfg["low_frac"]
        high = cfg["high_frac"]
        min_coop = cfg["min_frac_coop"]
        max_coop = cfg["max_frac_coop"]

        w = {
            "compare_numbers": 1.0,
            "reconcile_counts": 1.0,
            "translate_number": 1.2,
            "referential_signal": 1.4,
            "agreement_dialogue": 1.0,
            "describe_concept": 0.7,
            "narrative_chain": 0.7,
            "prediction_task": 0.6,
            "similarity_debate": 0.7,
            "role_assignment": 0.6,
            "misunderstanding_detection": 0.7,
            "definition_swap": 0.6,
            "action_reconstruction": 0.7,
            "property_attribution": 0.7,
            "verb_noun_compat": 0.7,
        }

        frac_exp = dist.get("explorer", 0.0)
        if frac_exp < low:
            w["describe_concept"] *= 2.2
            w["narrative_chain"] *= 2.0
            w["prediction_task"] *= 1.8
        elif frac_exp > high:
            w["describe_concept"] *= 0.5
            w["narrative_chain"] *= 0.5
            w["prediction_task"] *= 0.6

        frac_habit = dist.get("habit", 0.0)
        if frac_habit < low:
            w["action_reconstruction"] *= 2.0
            w["property_attribution"] *= 2.0
            w["verb_noun_compat"] *= 2.0
            w["reconcile_counts"] *= 1.5
        elif frac_habit > high:
            w["action_reconstruction"] *= 0.5
            w["property_attribution"] *= 0.5
            w["verb_noun_compat"] *= 0.6
            w["reconcile_counts"] *= 0.5

        frac_coop = dist.get("cooperator", 0.0)
        if frac_coop < min_coop:
            w["agreement_dialogue"] *= 2.0
            w["similarity_debate"] *= 1.7
            w["misunderstanding_detection"] *= 1.8
            w["role_assignment"] *= 1.6
            w["definition_swap"] *= 1.5
        elif frac_coop > max_coop:
            w["agreement_dialogue"] *= 0.5
            w["similarity_debate"] *= 0.6
            w["role_assignment"] *= 0.7
            w["definition_swap"] *= 0.7

        frac_lon = dist.get("loner", 0.0)
        if frac_lon > 0.15:
            w["agreement_dialogue"] *= 1.3
            w["misunderstanding_detection"] *= 1.3
            w["similarity_debate"] *= 1.2
            w["role_assignment"] *= 1.2
            w["prediction_task"] *= 0.7
            w["narrative_chain"] *= 0.7

        total = sum(w.values())
        if total <= 0:
            n = len(w)
            self.task_weights = {k: 1.0 / n for k in w}
        else:
            self.task_weights = {k: v / total for k, v in w.items()}

        if verbose:
            print("[HOMEOSTASIS] archetypes:", dist)
            print("[HOMEOSTASIS] task_weights:", self.task_weights)

    def _sample_task_type(self):
        items = list(self.task_weights.items())
        types, weights = zip(*items)
        r = random.random()
        cum = 0.0
        for t, w in zip(types, weights):
            cum += w
            if r <= cum:
                return t
        return types[-1]

    def _build_task_batch(self, num_tasks: int, gen: int):
        if FOCUSED_GROUNDING_EXPERIMENT:
            # Two numeral translations and two referential trials give each
            # convention enough repeated feedback to become useful.  This is
            # intentionally a temporary experimental curriculum, not a claim
            # that these are the only forms a mature language should support.
            ledger = getattr(self, "community_lexicon", None)
            numeric_ready = ledger is not None and len(ledger.numeric_conventions()) >= 8
            language_ready = (
                ledger is not None
                and len(ledger.numeric_conventions()) >= 3
                and len(ledger.referential_conventions()) >= 3
            )
            two_slot_ready = (
                ledger is not None
                and ledger.grammar_order("referent_quantity") is not None
            )
            action_ready = ledger is not None and len(ledger.action_conventions()) >= 2
            number_maintenance = numeric_ready and gen % 6 == 0
            action_maintenance = action_ready and gen % 7 == 0
            task_types = [
                "reconcile_counts",
                "translate_number" if number_maintenance else (
                    "translate_quantity" if numeric_ready else "translate_number"
                ),
                "referential_signal",
                (
                    "action_maintenance" if action_maintenance
                    else "compositional_action_signal" if action_ready
                    # Build a stable two-slot word order before asking agents
                    # to extend it with an action slot.  The staged curriculum
                    # makes grammatical agreement evidence cumulative rather
                    # than splitting it across arbitrary three-slot orders.
                    else "action_signal" if two_slot_ready
                    else "compositional_signal" if language_ready
                    else "referential_signal"
                ),
            ]
            task_types = task_types[:num_tasks]
            # A human word with a dictionary entry opens an additional,
            # inspectable semantic task.  It supplements rather than replaces
            # the established grounding curriculum.
            if (
                getattr(self, "human_token_memory", None)
                and gen % 2 == 0
                and self._human_dictionary_candidates()
            ):
                task_types.append("human_dictionary_link")
        else:
            self._update_task_weights_homeostasis(verbose=(gen % 20 == 0))
            task_types = [self._sample_task_type() for _ in range(num_tasks)]

        batch = []
        for ttype in task_types:
            generator = getattr(self, f"generate_{ttype}_task", None)
            if generator is None:
                generator = self.generate_numeric_compare_task

            task = generator()
            if task:
                batch.append(task)

        return batch

    # ---------------------------------------------------------
    #  Task generators
    # ---------------------------------------------------------
    def _human_dictionary_candidates(self):
        """Human-origin words that have useful, bounded dictionary evidence."""
        dictionary = getattr(self, "human_dictionary", None)
        origins = getattr(self, "human_token_origins", {}) or {}
        if dictionary is None:
            return []
        candidates = []
        for word, origin in origins.items():
            if origin != "human":
                continue
            relations = dictionary.semantic_relations(word)
            if relations and (relations["synonyms"] or relations["antonyms"]):
                candidates.append((word, relations))
        return candidates

    def generate_human_dictionary_link_task(self):
        """Ask an agent to identify a dictionary-supported human word link."""
        candidates = self._human_dictionary_candidates()
        if not candidates or not self.agents:
            return None

        counts = getattr(self, "human_dictionary_link_counts", {})
        words = [word for word, _ in candidates]
        active_topic = getattr(
            getattr(self, "community_conversation", None), "active_topic", None
        )
        # When a discussion has a subject, dictionary practice reinforces the
        # human term currently carrying that discussion.  It remains balanced
        # by the per-word count, so one topic cannot monopolise every task.
        weights = [
            (2.5 if word == active_topic else 1.0) / (1.0 + counts.get(word, 0))
            for word in words
        ]
        word, relations = random.choices(candidates, weights=weights, k=1)[0]
        relation = "synonym" if relations["synonyms"] else "antonym"
        accepted = relations["synonyms"] if relation == "synonym" else relations["antonyms"]
        accepted = accepted[:6]
        if not accepted:
            return None
        target = random.choice(accepted)
        distractors = [candidate for candidate in words if candidate != word and candidate not in accepted]
        options = list(dict.fromkeys([target] + distractors[:3]))
        random.shuffle(options)
        agent = random.choice(self.agents)
        counts[word] += 1
        return {
            "task_id": self._generate_task_id(),
            "task_type": "human_dictionary_link",
            "assigned_agents": [agent.id],
            "data": {
                "word": word,
                "relation": relation,
                "options": options,
                "accepted": accepted,
            },
            "responses": [],
        }

    def generate_describe_concept_task(self):
        if not self.agents:
            return None

        ag = random.choice(self.agents)
        all_tokens = list(getattr(ag.semantic, "vecs", {}).keys())
        target = random.choice(all_tokens) if all_tokens else "su"

        return {
            "task_id": self._generate_task_id(),
            "task_type": "describe_concept",
            "assigned_agents": [ag.id],
            "data": {"target_token": target},
            "responses": {}
        }

    def generate_narrative_chain_task(self):
        if not self.agents:
            return None

        ag = random.choice(self.agents)
        toks = list(getattr(ag.semantic, "vecs", {}).keys())
        start = random.choice(toks) if toks else "su"

        return {
            "task_id": self._generate_task_id(),
            "task_type": "narrative_chain",
            "assigned_agents": [ag.id],
            "data": {"start": start},
            "responses": {}
        }

    def generate_prediction_task(self):
        if not self.agents:
            return None

        ag = random.choice(self.agents)

        try:
            prefix = ag.produce_utterance()
        except Exception:
            prefix = "su tol"

        return {
            "task_id": self._generate_task_id(),
            "task_type": "prediction_task",
            "assigned_agents": [ag.id],
            "data": {"prefix": prefix},
            "responses": {}
        }

    def generate_similarity_debate_task(self):
        if not self.agents:
            return None

        ag = random.choice(self.agents)
        vecs = getattr(ag.semantic, "vecs", {})
        toks = list(vecs.keys())

        if len(toks) < 3:
            toks = ["su", "tol", "muk"]

        A, B, C = random.sample(toks, 3) if len(toks) >= 3 else ("su", "tol", "muk")

        return {
            "task_id": self._generate_task_id(),
            "task_type": "similarity_debate",
            "assigned_agents": [ag.id],
            "data": {"A": A, "B": B, "C": C},
            "responses": {}
        }

    def generate_role_assignment_task(self):
        if not self.agents:
            return None

        ag = random.choice(self.agents)

        vecs = getattr(ag.semantic, "vecs", {})
        events = list(vecs.keys()) or ["muk"]
        event = random.choice(events)

        return {
            "task_id": self._generate_task_id(),
            "task_type": "role_assignment",
            "assigned_agents": [ag.id],
            "data": {"event": event},
            "responses": {}
        }

    def generate_misunderstanding_detection_task(self):
        if len(self.agents) < 2:
            return None

        a, b = random.sample(self.agents, 2)

        try:
            utt = a.produce_utterance()
        except Exception:
            utt = "su tol rin"

        return {
            "task_id": self._generate_task_id(),
            "task_type": "misunderstanding_detection",
            "assigned_agents": [b.id],
            "data": {"utterance": utt, "partner_id": a.id},
            "responses": {}
        }

    def generate_definition_swap_task(self):
        if len(self.agents) < 2:
            return None

        a, b = random.sample(self.agents, 2)
        vecs = getattr(a.semantic, "vecs", {})
        toks = list(vecs.keys()) or ["su"]
        tok = random.choice(toks)

        try:
            partner_def = a.produce_utterance()
        except Exception:
            partner_def = tok + " su tol"

        return {
            "task_id": self._generate_task_id(),
            "task_type": "definition_swap",
            "assigned_agents": [b.id],
            "data": {"token": tok, "partner_definition": partner_def},
            "responses": {}
        }

    def generate_action_reconstruction_task(self):
        if not self.agents:
            return None

        ag = random.choice(self.agents)

        vecs = getattr(ag.semantic, "vecs", {})
        toks = list(vecs.keys()) or ["su", "tol", "muk"]

        if len(toks) < 2:
            a_tok, b_tok = "su", "tol"
        else:
            a_tok, b_tok = random.sample(toks, 2)

        return {
            "task_id": self._generate_task_id(),
            "task_type": "action_reconstruction",
            "assigned_agents": [ag.id],
            "data": {"from": a_tok, "to": b_tok},
            "responses": {}
        }

    def generate_property_attribution_task(self):
        if not self.agents:
            return None

        ag = random.choice(self.agents)

        vecs = getattr(ag.semantic, "vecs", {})
        toks = list(vecs.keys()) or ["su"]
        target = random.choice(toks)

        return {
            "task_id": self._generate_task_id(),
            "task_type": "property_attribution",
            "assigned_agents": [ag.id],
            "data": {"target_token": target},
            "responses": {}
        }

    def generate_verb_noun_compat_task(self):
        if not self.agents:
            return None

        ag = random.choice(self.agents)

        vecs = getattr(ag.semantic, "vecs", {})
        toks = list(vecs.keys()) or ["muk"]
        verb = random.choice(toks)

        return {
            "task_id": self._generate_task_id(),
            "task_type": "verb_noun_compat",
            "assigned_agents": [ag.id],
            "data": {"verb_token": verb},
            "responses": {}
        }

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

    # ------------------------------------------------------
    #  Task scorers
    # ------------------------------------------------------
    def score_task(self, task):
        ttype = task.get("task_type")
        scorer = getattr(self, f"score_{ttype}", None)
        if scorer:
            try:
                scorer(task)
            except Exception as e:
                print(f"[SCORER ERROR] {ttype}: {e}")

    def score_pref_align(self, task):
        responses = task.get("responses", [])
        if len(responses) != 2:
            return

        r1, r2 = responses
        if "choice" not in r1 or "choice" not in r2:
            return

        if r1["choice"] != r2["choice"]:
            return

        a_id = int(r1["agent_id"][1:])
        b_id = int(r2["agent_id"][1:])

        a = self.agent_by_id(a_id)
        b = self.agent_by_id(b_id)
        if not a or not b:
            return

        a.own_fitness += 0.2
        b.own_fitness += 0.2

        if hasattr(a, "adjust_trust"):
            a.adjust_trust(b_id, +0.05, channel=2)
            b.adjust_trust(a_id, +0.05, channel=2)

        task["evaluation"] = {
            "agreed": True,
            "agents": [a_id, b_id],
            "choice": r1["choice"],
        }

    def score_compare_numbers(self, task):
        gt = task.get("ground_truth", {})
        correct = gt.get("answer_phrase")
        if not correct:
            return

        for ag in self.agents:
            resp = ag.solved_tasks.get(task["task_id"])
            if not resp:
                continue

            if resp.get("answer") == correct:
                ag.own_fitness += 0.25
            else:
                ag.own_fitness -= 0.05

    def score_reconcile_counts(self, task):
        responses = task.get("responses", [])
        target = task.get("value")
        sources = task.get("signal_sources", {}) or {}
        views = task.get("views", {}) or {}
        correct = 0

        for r in responses:
            aid = int(r["agent_id"][1:])
            ag = self.agent_by_id(aid)
            if not ag:
                continue

            if r.get("normalized") == target:
                correct += 1
                phrase = views.get(aid, views.get(str(aid), ""))
                if phrase and len(phrase.split()) == 1:
                    self.community_lexicon.observe_numeric_success(target, phrase)
                ag.own_fitness += 0.35
                source = self.agent_by_id(sources.get(aid, sources.get(str(aid))))
                if source:
                    source.own_fitness += 0.15
                    ag.adjust_trust(source.id, +0.04, channel=4)
            else:
                ag.own_fitness -= 0.05
                phrase = views.get(aid, views.get(str(aid), ""))
                if phrase and len(phrase.split()) == 1:
                    ag.numeric_system.learn_digit_mapping(phrase, target)

        self._record_task_evaluation(task, correct=correct)

    def score_translate_number(self, task):
        self.score_reconcile_counts(task)

    def score_translate_quantity(self, task):
        self.score_reconcile_counts(task)
        correct = (task.get("evaluation", {}) or {}).get("correct", 0)
        base = task.get("numeric_base")
        if correct and base is not None:
            for _ in range(correct):
                self.community_lexicon.observe_base_success(base)

    def score_referential_signal(self, task):
        responses = task.get("responses", [])
        data = task.get("data", {}) or {}
        referent = data.get("referent")
        signal = data.get("signal")
        speaker = self.agent_by_id(data.get("speaker_id"))
        correct = 0

        for response in responses:
            try:
                listener_id = int(response["agent_id"][1:])
            except (KeyError, TypeError, ValueError):
                continue
            listener = self.agent_by_id(listener_id)
            if listener is None:
                continue

            if response.get("referent") == referent:
                correct += 1
                self.community_lexicon.observe_referential_success(referent, signal)
                listener.own_fitness += 0.45
                if speaker:
                    speaker.own_fitness += 0.20
                    listener.adjust_trust(speaker.id, +0.05, channel=4)
            else:
                listener.own_fitness -= 0.05
                # Grounded correction: retain the peer's signal for the world
                # referent, making the next encounter interpretable.
                listener.referent_lexicon[referent] = signal
                listener.vocab.add(signal)
                listener.semantic_system.ensure_vec(signal)
                listener._ensure_token_semantic(signal)

        self._record_task_evaluation(task, correct=correct)

    def score_action_signal(self, task):
        responses = task.get("responses", [])
        data = task.get("data", {}) or {}
        action = data.get("action")
        signal = data.get("signal")
        speaker = self.agent_by_id(data.get("speaker_id"))
        correct = 0

        for response in responses:
            try:
                listener_id = int(response["agent_id"][1:])
            except (KeyError, TypeError, ValueError):
                continue
            listener = self.agent_by_id(listener_id)
            if listener is None:
                continue
            if response.get("action") == action:
                correct += 1
                self.community_lexicon.observe_action_success(action, signal)
                listener.own_fitness += 0.45
                if speaker:
                    speaker.own_fitness += 0.20
                    listener.adjust_trust(speaker.id, +0.05, channel=4)
            else:
                listener.own_fitness -= 0.05
                listener.action_lexicon[action] = signal
                listener.vocab.add(signal)
                listener.semantic_system.ensure_vec(signal)
                listener._ensure_token_semantic(signal)

        self._record_task_evaluation(task, correct=correct)

    def score_human_dictionary_link(self, task):
        """Reward correct lookup and reinforce the discovered semantic edge."""
        data = task.get("data", {}) or {}
        word = data.get("word")
        relation = data.get("relation")
        accepted = set(data.get("accepted", []) or [])
        correct = 0
        for response in task.get("responses", []):
            try:
                agent_id = int(response["agent_id"][1:])
            except (KeyError, TypeError, ValueError):
                continue
            agent = self.agent_by_id(agent_id)
            related = response.get("related")
            if agent is None or related not in accepted:
                continue
            correct += 1
            agent.own_fitness += 0.20
            try:
                weight = +0.18 if relation == "synonym" else -0.14
                agent.semantic_system.link(word, related, weight)
            except Exception:
                pass
        self._record_task_evaluation(task, correct=correct)

    def score_compositional_signal(self, task):
        responses = task.get("responses", [])
        data = task.get("data", {}) or {}
        correct = 0
        for response in responses:
            if (
                response.get("referent") == data.get("referent")
                and response.get("value") == data.get("value")
            ):
                correct += 1
                agent = self.agent_by_id(int(response["agent_id"][1:]))
                if agent:
                    agent.own_fitness += 0.50
                self.community_lexicon.observe_grammar_success(
                    "referent_quantity", response.get("order"),
                )
        self._record_task_evaluation(task, correct=correct)

    def score_compositional_action_signal(self, task):
        responses = task.get("responses", [])
        data = task.get("data", {}) or {}
        correct = 0
        for response in responses:
            if (
                response.get("referent") == data.get("referent")
                and response.get("action") == data.get("action")
                and response.get("value") == data.get("value")
            ):
                correct += 1
                agent = self.agent_by_id(int(response["agent_id"][1:]))
                if agent:
                    agent.own_fitness += 0.70
                self.community_lexicon.observe_grammar_success(
                    "referent_action_number", response.get("order"),
                )
        self._record_task_evaluation(task, correct=correct)

    @staticmethod
    def _record_task_evaluation(task, correct=0):
        """Store comparable participation and correctness data for a task."""
        assigned = task.get("assigned_agents") or []
        attempted = task.get("attempted_by") or []
        responses = task.get("responses") or []
        answered_ids = {
            response.get("agent_id")
            for response in responses
            if isinstance(response, dict) and response.get("agent_id")
        }
        task["evaluation"] = {
            "assigned": len(assigned),
            "attempted": len(set(attempted)),
            "answered": len(answered_ids),
            "correct": correct,
        }

    def compact_numeric_overlays(self):
        """Keep agent-owned maps limited to genuine local alternatives."""
        for agent in self.agents:
            numeric = getattr(agent, "numeric_system", None)
            if numeric is not None and hasattr(numeric, "compact_against_community"):
                # Re-run normalisation after this generation's promotions so
                # a former private token cannot temporarily collide with a
                # newly public token for another digit.
                numeric.set_symbol_map(getattr(agent, "symbol_map", {}))
                numeric.compact_against_community()

    def score_agreement_dialogue(self, task):
        responses = task.get("responses", [])
        if len(responses) < 2:
            return

        proposals = [tuple(r.get("proposal", "").split()) for r in responses]
        if not proposals:
            return

        majority = max(set(proposals), key=proposals.count)
        frac = proposals.count(majority) / len(proposals)

        for r in responses:
            aid = int(r["agent_id"][1:])
            ag = self.agent_by_id(aid)
            if not ag:
                continue

            if tuple(r.get("proposal", "").split()) == majority:
                ag.own_fitness += 0.12 * frac

    def score_similarity_debate(self, task):
        responses = task.get("responses", [])
        if len(responses) < 2:
            return

        pairs = [tuple(r.get("chosen_pair", [])) for r in responses if r.get("chosen_pair")]
        if not pairs:
            return

        majority = max(set(pairs), key=pairs.count)
        frac = pairs.count(majority) / len(pairs)

        for r in responses:
            aid = int(r["agent_id"][1:])
            ag = self.agent_by_id(aid)
            if not ag:
                continue

            if tuple(r.get("chosen_pair", [])) == majority:
                ag.own_fitness += 0.10 * frac

    def score_definition_swap(self, task):
        responses = task.get("responses", [])
        if len(responses) < 2:
            return

        token_sets = [
            set(r.get("utterance", "").split())
            for r in responses
            if r.get("utterance")
        ]

        if len(token_sets) < 2:
            return

        overlap = set.intersection(*token_sets)
        score = min(len(overlap) / 4.0, 1.0)

        for r in responses:
            aid = int(r["agent_id"][1:])
            ag = self.agent_by_id(aid)
            if ag:
                ag.own_fitness += 0.08 * score

    def score_misunderstanding_detection(self, task):
        responses = task.get("responses", [])
        if not responses:
            return

        base_tokens = set(task.get("data", {}).get("utterance", "").split())

        for r in responses:
            aid = int(r["agent_id"][1:])
            ag = self.agent_by_id(aid)
            if not ag:
                continue

            out_tokens = set(r.get("interpretation", "").split())
            overlap = base_tokens & out_tokens

            ag.own_fitness += 0.05 + 0.02 * len(overlap)

    def score_role_assignment(self, task):
        responses = task.get("responses", [])
        if len(responses) < 2:
            return

        doers = [r.get("doer") for r in responses if r.get("doer")]
        receivers = [r.get("receiver") for r in responses if r.get("receiver")]

        doer_consensus = len(set(doers)) == 1
        recv_consensus = len(set(receivers)) == 1

        for r in responses:
            aid = int(r["agent_id"][1:])
            ag = self.agent_by_id(aid)
            if not ag:
                continue

            if doer_consensus:
                ag.own_fitness += 0.07
            if recv_consensus:
                ag.own_fitness += 0.07

    def score_prediction_task(self, task):
        responses = task.get("responses", [])
        if len(responses) < 2:
            return

        preds = [r.get("prediction") for r in responses if r.get("prediction")]
        if not preds:
            return

        majority = max(set(preds), key=preds.count)
        frac = preds.count(majority) / len(preds)

        for r in responses:
            aid = int(r["agent_id"][1:])
            ag = self.agent_by_id(aid)
            if ag and r.get("prediction") == majority:
                ag.own_fitness += 0.08 * frac
