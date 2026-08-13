import random

from evolution.coordinator_settings import FOCUSED_GROUNDING_EXPERIMENT


class CoordinatorTaskPlanningMixin:
    def _generate_task_id(self):
        tid = f"T{self.next_task_id:04d}"
        self.next_task_id += 1
        return tid

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
