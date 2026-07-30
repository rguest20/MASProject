import random

from evolution.coordinator_settings import FOCUSED_GROUNDING_EXPERIMENT


class CoordinatorTaskScoringMixin:
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
