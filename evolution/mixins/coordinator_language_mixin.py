import math
import random
import re
from collections import Counter, defaultdict

from agents.cognition.semantic_utils import add, scale, cos_sim
from evolution.coordinator_settings import (
    REFERENTIAL_BONUS,
    PHASE2_LR,
    REWARD_EPS,
    REWARD_TEMP,
    TEACH_PAIRS_PER_PASS,
    TEACH_ACC_TEMP,
    TEACH_TOP_FRACTION,
    TEACH_PROB,
    IMPROVE_ONLY_TEACH,
    OUTCOME_SCALE,
)
from evolution.programs import run_program, mutate_program


class CoordinatorLanguageMixin:
    """
    Language, teaching, and dialogue utilities factored out of the core
    Coordinator.  These methods manipulate agent communication without
    needing to live inside the already large coordinator module.
    """

    _READING_TOKEN = re.compile(r"[a-z][a-z'-]{0,31}", re.IGNORECASE)

    @classmethod
    def _reading_tokens(cls, sentence):
        return cls._READING_TOKEN.findall(str(sentence or "").lower())[:30]

    def _conversation_is_quiet(self):
        """A quiet turn has no unanswered Ryan prompt waiting for a reply."""
        bridge = getattr(self, "community_conversation", None)
        if bridge is None:
            return True
        try:
            transcript = bridge._transcript()
            return transcript is None or bridge._latest_unanswered_prompt(transcript) is None
        except Exception:
            return True

    @staticmethod
    def _cosine(left, right):
        if not left or not right:
            return 0.0
        size = min(len(left), len(right))
        dot = sum(float(left[index]) * float(right[index]) for index in range(size))
        left_norm = math.sqrt(sum(float(value) ** 2 for value in left[:size]))
        right_norm = math.sqrt(sum(float(value) ** 2 for value in right[:size]))
        if left_norm <= 1e-9 or right_norm <= 1e-9:
            return 0.0
        return dot / (left_norm * right_norm)

    @staticmethod
    def _subtract_vectors(left, right):
        return [float(a) - float(b) for a, b in zip(left or [], right or [])]

    @staticmethod
    def _contains_span(sentence, candidate):
        """Whether a candidate is only a contiguous excerpt of a source."""
        width = len(candidate)
        return bool(width and any(
            tuple(sentence[index:index + width]) == tuple(candidate)
            for index in range(max(0, len(sentence) - width + 1))
        ))

    def _sentence_vector(self, agent, tokens):
        semantic = getattr(agent, "semantic_system", None)
        if semantic is None or not tokens:
            return []
        vectors = []
        for token in tokens:
            try:
                vectors.append(semantic.ensure_vec(token))
            except Exception:
                continue
        if not vectors:
            return []
        size = min(len(vector) for vector in vectors)
        if not size:
            return []
        return [
            sum(float(vector[index]) for vector in vectors) / len(vectors)
            for index in range(size)
        ]

    def _sentence_link_agreement(self, left_tokens, right_tokens):
        """How similarly members represent the move from one sentence to next."""
        directions = []
        for agent in self.agents:
            left = self._sentence_vector(agent, left_tokens)
            right = self._sentence_vector(agent, right_tokens)
            direction = self._subtract_vectors(right, left)
            if sum(value * value for value in direction) > 1e-9:
                directions.append(direction)
        if len(directions) < 2:
            return 0.0
        similarities = [
            self._cosine(left, right)
            for index, left in enumerate(directions)
            for right in directions[index + 1:]
        ]
        return sum(similarities) / max(1, len(similarities))

    def _reading_proposal(self, agent, source_tokens):
        """Compose a short candidate using only locally read continuations."""
        transitions = getattr(self, "reading_sentence_transitions", {}) or {}
        pool = getattr(self, "reading_sentence_tokens", {}) or {}
        if not transitions or not pool:
            return []
        starts = [source_tokens[0]] if source_tokens and source_tokens[0] in pool else []
        if not starts:
            starts = list((getattr(self, "reading_sentence_start_tokens", {}) or {}).keys())
        if not starts:
            return []
        preferences = getattr(agent, "utter_bias", {}).get("symbol_preferences", {})
        start_weights = [
            max(0.05, float(pool.get(token, 0.05)))
            * (0.5 + float(preferences.get(token, 0.2)))
            for token in starts
        ]
        words = [random.choices(starts, weights=start_weights, k=1)[0]]
        target_length = random.randint(4, 8)
        endings = getattr(self, "reading_sentence_end_tokens", {}) or {}
        while len(words) < 10:
            current = words[-1]
            if len(words) >= target_length and endings.get(current, 0) > 0:
                break
            candidates = list((transitions.get(current, {}) or {}).keys())
            candidates = [word for word in candidates if word in pool]
            if not candidates:
                break
            unseen = [word for word in candidates if words.count(word) < 2]
            if unseen:
                candidates = unseen
            weights = [
                max(0.05, float(transitions[current].get(word, 0.05)))
                * (0.5 + float(preferences.get(word, 0.2)))
                for word in candidates
            ]
            words.append(random.choices(candidates, weights=weights, k=1)[0])
        # A sentence boundary is evidence supplied by the corpus itself.  It
        # avoids rewarding an arbitrary truncation such as ``... such a``.
        return words if len(words) >= 4 and endings.get(words[-1], 0) > 0 else []

    def _practice_reading_intent(self, passage_tokens):
        """Retain a proposal only if it beats real alternative continuations.

        Each listener compares directions inside its own semantic space;
        agent vector coordinates are private, so direct cross-agent cosine is
        not a valid measure of shared meaning. Agreement is the fraction of
        listeners for whom this proposal fits the current next sentence better
        than several previously read next sentences.
        """
        metrics = self.community_learning_metrics
        if len(passage_tokens) < 2 or not self.agents:
            return None
        source, destination = passage_tokens[0], passage_tokens[1]
        contrasts = []
        for record in reversed(getattr(self, "reading_context_directions", ())):
            alternative = list(record.get("right", ()))
            if alternative and alternative != destination and alternative not in contrasts:
                contrasts.append(alternative)
            if len(contrasts) >= 3:
                break
        # Early passages build a comparison set. Without real alternatives,
        # a positive cosine is self-validating rather than meaningful.
        if len(contrasts) < 2:
            return None

        proposals = []
        source_forms = {tuple(tokens) for tokens in passage_tokens}
        for agent in self.agents:
            proposal = self._reading_proposal(agent, source)
            copied_span = any(self._contains_span(tokens, proposal) for tokens in passage_tokens)
            if len(proposal) >= 4 and tuple(proposal) not in source_forms and not copied_span:
                proposals.append(proposal)
        metrics["intent_proposals"] = len(proposals)
        if not proposals:
            return None

        best = None
        for proposal in proposals:
            target_scores = []
            margins = []
            for voter in self.agents:
                source_vector = self._sentence_vector(voter, source)
                intended_direction = self._subtract_vectors(
                    self._sentence_vector(voter, destination), source_vector
                )
                proposed_direction = self._subtract_vectors(
                    self._sentence_vector(voter, proposal), source_vector
                )
                target_score = self._cosine(proposed_direction, intended_direction)
                contrast_scores = [
                    self._cosine(
                        proposed_direction,
                        self._subtract_vectors(
                            self._sentence_vector(voter, alternative), source_vector
                        ),
                    )
                    for alternative in contrasts
                ]
                target_scores.append(target_score)
                margins.append(target_score - max(contrast_scores, default=0.0))
            votes = sum(
                target >= 0.12 and margin >= 0.10
                for target, margin in zip(target_scores, margins)
            )
            mean_target = sum(target_scores) / max(1, len(target_scores))
            mean_margin = sum(margins) / max(1, len(margins))
            candidate = (votes, mean_margin, mean_target, proposal)
            if best is None or candidate[:3] > best[:3]:
                best = candidate

        votes, margin, similarity, proposal = best
        agreement = votes / max(1, len(self.agents))
        metrics["intent_agreement"] = agreement
        metrics["intent_margin"] = margin
        metrics["intent_target_similarity"] = similarity
        if agreement < 0.65 or margin < 0.10 or similarity < 0.12:
            return None

        # This is an internal semantic-practice success, not yet a polished
        # human utterance.  Do not let one Markov-style recombination enter
        # Ryan-facing production: it can be directionally useful while still
        # sounding rough.  The agents retain it privately; external English
        # remains grounded in Ryan's wording or explicit human feedback.
        for agent in self.agents:
            try:
                agent._observe_language_tokens(proposal, gain=0.08)
                agent.state_event("reading_intent_agreement")
                agent.relieve_community_discomfort(0.03)
            except Exception:
                continue

        metrics["intent_successes"] = 1
        state = self.community_learning_state
        state["intent_success_total"] += 1
        return {
            "sentence": " ".join(proposal) + ".",
            "agreement": agreement,
            "similarity": similarity,
            "margin": margin,
        }

    def _learn_from_reading_passage(self, passage):
        """Let every member observe a tiny shared passage in local context."""
        metrics = self.community_learning_metrics
        tokenised = [self._reading_tokens(sentence) for sentence in passage]
        tokenised = [tokens for tokens in tokenised if tokens]
        if not tokenised:
            return None

        new_tokens = set()
        for tokens in tokenised:
            for token in tokens:
                if token not in self.reading_token_memory:
                    new_tokens.add(token)
                self.reading_token_memory[token] += 1
                self.reading_sentence_tokens[token] += 1
            for left, right in zip(tokens, tokens[1:]):
                self.reading_sentence_transitions[left][right] += 1
            self.reading_sentence_start_tokens[tokens[0]] += 1
            self.reading_sentence_end_tokens[tokens[-1]] += 1

        for agent in self.agents:
            try:
                all_tokens = [token for sentence in tokenised for token in sentence]
                agent.vocab.update(all_tokens)
                agent.recent_tokens.extend(all_tokens)
                agent.recent_tokens = agent.recent_tokens[-64:]
                preferences = agent.utter_bias["symbol_preferences"]
                for token in all_tokens:
                    preferences[token] = max(float(preferences.get(token, 0.2)), 0.45)
                    agent.semantic_system.ensure_vec(token)
                    agent._ensure_token_semantic(token)
                # The sentence's own local order is the learning signal.
                # We intentionally do not send it to CommunityWorldModel.
                for tokens in tokenised:
                    agent._observe_language_tokens(tokens, gain=0.12)
                agent.state_event("reading_progress")
                agent.relieve_community_discomfort(0.015)
            except Exception:
                continue

        link_agreement = 0.0
        for left, right in zip(tokenised, tokenised[1:]):
            link_agreement = self._sentence_link_agreement(left, right)
            self.reading_context_directions.append({
                "generation": self.generation_index,
                "left": tuple(left),
                "right": tuple(right),
                "agreement": link_agreement,
            })
        metrics.update(
            reading_sentences=len(tokenised),
            reading_new_tokens=len(new_tokens),
            reading_link_agreement=link_agreement,
        )
        state = self.community_learning_state
        state["sentences_read_total"] += len(tokenised)
        state["stagnation_streak"] = 0
        intent = self._practice_reading_intent(tokenised)
        try:
            with open(self.reading_log_path, "a", encoding="utf-8") as log:
                log.write(f"Gen {self.generation_index} shared reading\n")
                for sentence in passage:
                    log.write(f"  Read: {sentence}\n")
                log.write(
                    f"  New tokens={len(new_tokens)} "
                    f"link-agreement={link_agreement:.3f}\n"
                )
                if intent is not None:
                    log.write(
                        f"  Intent practice (internal): {intent['sentence']} "
                        f"agreement={intent['agreement']:.3f} "
                        f"direction={intent['similarity']:.3f} "
                        f"margin={intent['margin']:.3f}\n"
                    )
                else:
                    log.write("  Intent practice: no consensus candidate\n")
        except OSError:
            pass
        return {"new_tokens": len(new_tokens), "intent": intent}

    def run_quiet_reading_cycle(self):
        """Read only after quiet stretches, with faster relief under pressure."""
        metrics = self.community_learning_metrics
        for key in (
            "reading_sentences", "reading_new_tokens", "reading_link_agreement",
            "intent_proposals", "intent_agreement", "intent_margin",
            "intent_target_similarity", "intent_successes",
        ):
            metrics[key] = 0.0 if "agreement" in key else 0
        state = self.community_learning_state
        quiet = self._conversation_is_quiet()
        state["quiet_streak"] = state["quiet_streak"] + 1 if quiet else 0
        if not quiet:
            state["stagnation_streak"] = 0

        room = getattr(self, "community_reading", None)
        discomfort = float(getattr(self, "community_discomfort", 0.0))
        cadence = 2 if discomfort >= 0.50 else 4
        due = (
            quiet
            and state["quiet_streak"] >= 3
            and self.generation_index - state["last_read_generation"] >= cadence
            and room is not None
            and room.available
        )
        if due:
            passage = room.next_passage(sentences=2)
            result = self._learn_from_reading_passage(passage)
            if result is not None:
                state["last_read_generation"] = self.generation_index
        elif quiet:
            state["stagnation_streak"] += 1

        metrics.update(
            quiet_streak=state["quiet_streak"],
            stagnation_streak=state["stagnation_streak"],
            reading_total_sentences=state["sentences_read_total"],
        )

    def update_community_discomfort(self):
        """Convert stagnation and repeated errors into a bounded learning drive."""
        state = self.community_learning_state
        attempted = 0
        correct = 0
        for task in getattr(self, "active_tasks", []) or []:
            evaluation = task.get("evaluation", {}) or {}
            attempted += int(evaluation.get("attempted", 0))
            correct += int(evaluation.get("correct", 0))
        wrong_rate = max(0.0, (attempted - correct) / attempted) if attempted else 0.0
        state["wrong_attempt_ema"] = (
            (0.75 * float(state["wrong_attempt_ema"])) + (0.25 * wrong_rate)
        )
        stagnation = max(0.0, (float(state["stagnation_streak"]) - 1.0) / 5.0)
        error_pressure = max(0.0, (float(state["wrong_attempt_ema"]) - 0.20) / 0.80)
        relief = min(
            0.06,
            0.01 * int(self.community_learning_metrics.get("reading_new_tokens", 0)),
        )
        if self.community_learning_metrics.get("intent_successes", 0):
            relief += 0.03
        target = min(1.0, (0.55 * stagnation) + (0.45 * error_pressure))
        unsettled = (0.78 * float(self.community_discomfort)) + (0.22 * target)
        self.community_discomfort = max(0.0, min(1.0, unsettled * (1.0 - relief)))
        for agent in self.agents:
            try:
                agent.receive_community_discomfort(
                    self.community_discomfort,
                    stagnation=stagnation,
                    errors=error_pressure,
                )
            except Exception:
                continue
        self.community_learning_metrics.update(
            discomfort=self.community_discomfort,
            wrong_attempt_rate=wrong_rate,
            quiet_streak=state["quiet_streak"],
            stagnation_streak=state["stagnation_streak"],
            reading_total_sentences=state["sentences_read_total"],
        )

    def gossip_exchange(self):
        agents = self.agents
        if len(agents) < 2:
            return

        teacher, student = random.sample(agents, 2)
        if random.random() < 0.3:
            if teacher.semantic["vecs"]:
                word = random.choice(list(teacher.semantic["vecs"].keys()))
                teacher.teaching_system.teach_student(student, word)

        for sender in agents:
            rep_list = sender.export_reputation(top_k=5) or []
            conf = float(getattr(sender, "reputation_strength", 0.012))

            num_receivers = random.randint(1, 3)
            receivers = random.sample(agents, min(num_receivers, len(agents)))

            for recv in receivers:
                if recv.id == sender.id:
                    continue

                recv.adjust_trust(
                    target_id=sender.id,
                    amount=min(conf, 0.02),
                    channel=2
                )

                for pid, rep in rep_list:
                    if pid == recv.id or pid == sender.id:
                        continue
                    influence = 0.05 * conf * float(rep)
                    if influence == 0.0:
                        continue
                    recv.adjust_trust(
                        target_id=pid,
                        amount=min(influence, 0.015),
                        channel=3
                    )

                if sender.semantic["vecs"]:
                    shared_word = random.choice(list(sender.semantic["vecs"].keys()))

                    recv.observe_utterance(
                        shared_word,
                        gain_scale=0.25
                    )

    def semantic_teaching_phase(self):
        if not self.agents:
            return

        pairs = min(TEACH_PAIRS_PER_PASS, len(self.agents))
        for _ in range(pairs):
            teacher, student = random.sample(self.agents, 2)

            if teacher.teaching_system.teaching_willingness(student.id) < 0.1 and random.random() < 0.85:
                continue

            bundle = teacher.export_semantic_bundle(max_keys=3)
            if not bundle:
                continue

            pred = student.attempt_prediction(bundle)
            if pred is None:
                continue

            toks = bundle.get("tokens", []) or []
            vecs = teacher.semantic["vecs"]
            gt = None
            try:
                if toks:
                    cols = [vecs[t] for t in toks if t in vecs]
                    if not cols:
                        continue
                    accv = cols[0][:]
                    for v in cols[1:]:
                        accv = add(accv, v)
                    gt = scale(accv, 1.0 / len(cols))
                else:
                    continue
            except Exception:
                continue

            try:
                acc = cos_sim(pred, gt)
            except Exception:
                acc = 0.0

            if TEACH_ACC_TEMP != 1.0 and acc is not None:
                acc = math.copysign(abs(acc) ** TEACH_ACC_TEMP, acc)

            reward = max(-1.0, min(1.0, float(acc)))
            if reward > 0:
                k = 0.10 * reward
            else:
                k = 0.02 * reward

            teacher.apply_teaching_reward(student.id, reward)
            student.apply_learning_reward(teacher.id, reward)

            k = 0.06 * reward
            teacher.update_trust_channels(student.id, reward)
            student.update_trust_channels(teacher.id, reward)

            student.adjust_trust(
                teacher.id,
                amount=max(-0.03, min(0.03, k)),
                channel=4
            )
            teacher.adjust_trust(
                student.id,
                amount=max(-0.02, min(0.02, k * 0.5)),
                channel=3
            )

    def language_feedback(self):
        tots = [a.total_fitness for a in self.agents]
        mmin, mmax = min(tots), max(tots)
        span = (mmax - mmin) if (mmax - mmin) > REWARD_EPS else 1.0

        base_r = {}
        for a in self.agents:
            x01 = (a.total_fitness - mmin) / span
            if REWARD_TEMP != 1.0:
                x01 = x01 ** REWARD_TEMP
            base_r[a.id] = (x01 * 2.0) - 1.0

        def jaccard(a_tokens, b_tokens):
            A, B = set(a_tokens), set(b_tokens)
            if not A or not B:
                return 0.0
            return len(A & B) / len(A | B)

        for listener in self.agents:
            for speaker_id, utt in self.last_utterances.items():
                if listener.id == speaker_id:
                    continue

                r = base_r.get(speaker_id, 0.0)

                speaker = next((x for x in self.agents if x.id == speaker_id), None)
                if speaker is not None:
                    utt_tokens = utt.split()
                    act_tokens = getattr(speaker, "_last_tokens", []) or []
                    r += REFERENTIAL_BONUS * jaccard(utt_tokens, act_tokens)

                r = max(-1.0, min(1.0, r))

                speaker = next((x for x in self.agents if x.id == speaker_id), None)
                if speaker is not None:
                    listener.numeric_system._maybe_learn_numeric_from(speaker)

                listener.learn_from_feedback(utt, reward=r, lr=PHASE2_LR)

        uniq_utts = len({u for u in self.last_utterances.values()})
        if uniq_utts <= 2:
            for a in self.agents:
                try:
                    a.lm.decay(rate=0.99)
                except Exception:
                    pass

    def reward_penalty_phase(self):
        for agent in self.agents:
            net_outcome = sum(v for (_, v) in agent.interaction_memory[-10:]) if hasattr(agent, "interaction_memory") else 0.0
            reward_signal = max(-1.0, min(1.0, net_outcome * 0.1))

            if reward_signal > 0:
                agent.energy += 0.05 * reward_signal
                agent.trust_bias = getattr(agent, "trust_bias", 0.0) + 0.01 * reward_signal
                agent.own_fitness += reward_signal
            elif reward_signal < 0:
                agent.energy -= 0.05 * abs(reward_signal)
                agent.trust_bias = getattr(agent, "trust_bias", 0.0) - 0.01 * abs(reward_signal)
                for pid in agent.social_memory.keys():
                    agent.adjust_trust(pid, amount=-0.005 * abs(reward_signal), channel=4)

        for agent in self.agents:
            # Keep interaction feedback as a small metabolic friction.  The
            # former 1–4% per-generation loss overwhelmed all recovery and
            # repeatedly replaced agents before learned language could persist.
            agent.energy *= random.uniform(0.998, 1.0)
            agent.energy = min(agent.energy, 100.0)

    def communicate(self, speaker, listener, utterance):
        shared = utterance in listener.utterance_memory["usage_count"]
        coop_avg = (speaker.traits["cooperation_weight"] + listener.traits["cooperation_weight"]) / 2
        success = shared and random.random() < coop_avg

        if success:
            delta_e = 0.5
            speaker.energy = min(100.0, speaker.energy + delta_e)
            listener.energy = min(100.0, listener.energy + delta_e)

            speaker.traits["trust_threshold"] = max(
                0.0, speaker.traits["trust_threshold"] - 0.03
            )
            listener.traits["trust_threshold"] = max(
                0.0, listener.traits["trust_threshold"] - 0.03
            )

            speaker.adjust_trust(listener.id, amount=+0.02, channel=2)
            listener.adjust_trust(speaker.id, amount=+0.02, channel=2)
            speaker.adjust_trust(listener.id, amount=+0.01, channel=3)
            listener.adjust_trust(speaker.id, amount=+0.01, channel=3)
        else:
            delta_e = 0.3
            speaker.energy = max(0.0, speaker.energy - delta_e)
            listener.energy = max(0.0, listener.energy - delta_e)
            speaker.traits["trust_threshold"] = min(
                1.0, speaker.traits["trust_threshold"] + 0.005
            )
            listener.traits["trust_threshold"] = min(
                1.0, listener.traits["trust_threshold"] + 0.005
            )

            speaker.adjust_trust(listener.id, amount=-0.01, channel=4)
            listener.adjust_trust(speaker.id, amount=-0.01, channel=4)

            old = speaker.utterance_memory["associations"].get(utterance, 0.0)
            speaker.utterance_memory["associations"][utterance] = old * 0.9

    @staticmethod
    def crossover_program(prog_a, prog_b):
        try:
            len_a, len_b = len(prog_a), len(prog_b)
            if len_a == 0 or len_b == 0:
                return prog_a[:] if len_a >= len_b else prog_b[:]
            cut_a = random.randrange(len_a)
            cut_b = random.randrange(len_b)
            child = prog_a[:cut_a] + prog_b[cut_b:]
            if len(child) == 0:
                child = (prog_a if random.random() < 0.5 else prog_b)[:]
            return child
        except Exception:
            return prog_a[:]

    def teaching_phase(self):
        ranked = sorted(self.agents, key=lambda a: a.total_fitness, reverse=True)
        top_n = max(1, int(len(self.agents) * TEACH_TOP_FRACTION))
        teachers = ranked[:top_n]

        for teacher in teachers:
            if random.random() > TEACH_PROB:
                continue

            candidates = [a for a in self.agents if a.id != teacher.id]
            if not candidates:
                continue

            weights = []
            for c in candidates:
                will = max(0.0, teacher.teaching_system.teaching_willingness(c.id))
                weights.append(0.05 + will)

            student = random.choices(candidates, weights=weights)[0]

            student.teaching_attempted = True
            teacher.teaching_attempted = True

            if teacher.teaching_system.teaching_willingness(student.id) < teacher.traits.get("trust_threshold", 0.3):
                continue

            new_prog = self.crossover_program(student.program, teacher.program)
            new_prog = mutate_program(new_prog)

            if IMPROVE_ONLY_TEACH:
                old_fit = student.own_fitness
                try:
                    tmp_fit = run_program(new_prog)
                except Exception:
                    tmp_fit = -1e9
                if tmp_fit > old_fit:
                    student.program = new_prog
                    outcome = (tmp_fit - old_fit) * OUTCOME_SCALE
                    student.remember_interaction(teacher.id, outcome=outcome, gen_index=self.generation_index)
                    teacher.remember_interaction(student.id, outcome=outcome * 0.5, gen_index=self.generation_index)
            else:
                student.program = new_prog

            bundle = teacher.export_semantic_bundle(max_keys=1)
            if not bundle:
                continue

            word = bundle["tokens"][0]
            expected_vec = bundle["vecs"][word]

            teach_reward, teach_penalty, sim = teacher.teaching_system.teach_student(
                student, word, current_gen=self.generation_index
            )

            eval_reward = student.teaching_system.evaluate_teaching(
                teacher.id, word, expected_vec
            )

            teacher.apply_teaching_reward(student.id, eval_reward)

            try:
                self.world.append_text(
                    "/notes.txt",
                    f"[TeachPhase] A{teacher.id}->{student.id} "
                    f"trust={teacher.teaching_system.teaching_willingness(student.id):.2f} "
                    f"word={word} sim={sim:.2f} "
                    f"teachR={teach_reward:.2f} evalR={eval_reward:.2f}\n"
                )
            except Exception:
                pass

    def run_language_phase(self):
        self.last_utterances = {}

        for a in self.agents:
            a.challenge_guess = random.randint(0, 4)
            a.challenge_system = self.challenge

        self.challenge.new_challenge()
        self.challenge.assign_liars(self.agents)

        for a in self.agents:
            try:
                # Broadcasts should rehearse the language that agents can
                # actually use together.  Before conventions exist we retain
                # exploratory speech; afterwards a broadcast is a compact,
                # interpretable practice act rather than random token salad.
                act = self._grounded_dialogue_act(a, listener=None)
                utt = act["signal"] if act is not None else a.produce_utterance()
                self.last_utterances[a.id] = utt
                if a.id in self.agent_apis:
                    self.agent_apis[a.id].append_text("/notes.txt", f"A{a.id}: {utt}\n", scope="world")
            except Exception:
                pass

        ids = list(self.last_utterances.keys())
        for _ in range(len(ids)):
            if len(ids) < 2:
                break
            speaker_id, listener_id = random.sample(ids, 2)
            speaker = next(a for a in self.agents if a.id == speaker_id)
            listener = next(a for a in self.agents if a.id == listener_id)
            utt = self.last_utterances[speaker_id]
            self.communicate(speaker, listener, utt)

        for a in self.agents:
            try:
                a.language_world_ingest_step()
            except Exception:
                pass

        self.semantic_teaching_phase()
        self.gossip_exchange()
        self.semantic_teaching_phase()

    def _grounded_dialogue_act(self, speaker, listener=None):
        """Create one short, interpretable social-language act.

        The protocol deliberately uses only the conventions agents have
        already earned through grounded tasks.  Free conversation therefore
        rehearses a shared language without silently creating a new oracle
        channel.  As the inventory grows it progresses from numerals, to
        referent-number descriptions, to referent-action-number requests.
        """
        ledger = getattr(self, "community_lexicon", None)
        if ledger is None:
            return None

        numeric = ledger.numeric_conventions()
        referential = ledger.referential_conventions()
        actions = ledger.action_conventions()
        if not numeric and not referential:
            return None

        teaching_drive = float(getattr(speaker, "traits", {}).get("teaching_drive", 0.0))
        learner_curiosity = float(getattr(listener, "traits", {}).get("curiosity", 0.0)) if listener else 0.0
        intent = "teach" if listener and random.random() < (0.15 + 0.35 * teaching_drive + 0.15 * learner_curiosity) else "practice"

        if numeric and referential and actions:
            order = ledger.grammar_order("referent_action_number")
            if order:
                referent, referent_token = random.choice(list(referential.items()))
                action, action_token = random.choice(list(actions.items()))
                value, number_token = random.choice(list(numeric.items()))
                tokens = {
                    "referent": referent_token,
                    "action": action_token,
                    "number": number_token,
                }
                return {
                    "intent": intent,
                    "kind": "compositional_action_signal",
                    "signal": " ".join(tokens[role] for role in order),
                    "meaning": {"referent": referent, "action": action, "value": value},
                }

        if numeric and referential:
            order = ledger.grammar_order("referent_quantity")
            if order:
                referent, referent_token = random.choice(list(referential.items()))
                value, number_token = random.choice(list(numeric.items()))
                tokens = {"referent": referent_token, "number": number_token}
                return {
                    "intent": intent,
                    "kind": "compositional_signal",
                    "signal": " ".join(tokens[role] for role in order),
                    "meaning": {"referent": referent, "value": value},
                }

        if numeric:
            value, token = random.choice(list(numeric.items()))
            return {
                "intent": intent,
                "kind": "number_practice",
                "signal": token,
                "meaning": {"value": value},
            }

        referent, token = random.choice(list(referential.items()))
        return {
            "intent": intent,
            "kind": "referent_practice",
            "signal": token,
            "meaning": {"referent": referent},
        }

    def _human_token_practice_act(self, speaker, listener):
        """Attach one human token to a valid community message for rehearsal."""
        token_counts = getattr(self, "human_token_memory", {}) or {}
        origins = getattr(self, "human_token_origins", {}) or {}
        token_counts = {
            token: count for token, count in token_counts.items()
            if origins.get(token) == "human"
        }
        if not token_counts:
            return None
        core = self._grounded_dialogue_act(speaker, listener)
        if core is None:
            return None
        token = self._choose_human_practice_token(token_counts)
        if token is None:
            return None
        return {
            "intent": "human_practice",
            "kind": "human_token_practice",
            "signal": f"{core['signal']} {token}",
            "meaning": {**core.get("meaning", {}), "human_token": token},
            "core": core,
        }

    def _choose_human_practice_token(self, token_counts):
        """Rotate through human words instead of repeatedly choosing a tie."""
        candidates = sorted(token_counts)
        if not candidates:
            return None

        recent = getattr(self, "human_practice_recent", None)
        if recent is None:
            from collections import deque
            self.human_practice_recent = deque(maxlen=8)
            recent = self.human_practice_recent
        practice_counts = getattr(self, "human_practice_counts", None)
        if practice_counts is None:
            from collections import Counter
            self.human_practice_counts = Counter()
            practice_counts = self.human_practice_counts

        # Do not repeat the last selected word while another human word is
        # available.  The short history further spreads practice across a
        # prompt's vocabulary, while weighted choice retains some variety.
        available = [token for token in candidates if token not in recent]
        if not available:
            available = [token for token in candidates if token != (recent[-1] if recent else None)]
        if not available:
            available = candidates
        weights = [
            (1.0 / (1.0 + practice_counts[token])) * (1.0 + min(3, token_counts[token]) * 0.10)
            for token in available
        ]
        token = random.choices(available, weights=weights, k=1)[0]
        practice_counts[token] += 1
        recent.append(token)
        return token

    def _social_dialogue_act(self, speaker, listener):
        """Mostly use grounded grammar; occasionally rehearse human input."""
        if getattr(self, "human_token_memory", None) and random.random() < 0.18:
            practice = self._human_token_practice_act(speaker, listener)
            if practice is not None:
                return practice
        return self._grounded_dialogue_act(speaker, listener)

    def _interpret_grounded_dialogue_act(self, listener, act):
        """Return whether a listener recovered the intended grounded meaning."""
        if act is None:
            return False
        kind = act.get("kind")
        signal = act.get("signal", "")
        meaning = act.get("meaning", {})
        task_system = getattr(listener, "task_system", None)

        try:
            if kind == "human_token_practice":
                core = act.get("core") or {}
                token = meaning.get("human_token")
                if not token or token not in listener.vocab:
                    return False
                return self._interpret_grounded_dialogue_act(listener, core)
            if kind == "compositional_action_signal" and task_system is not None:
                response = task_system._solve_compositional_action_signal({"data": {"signal": signal}})
                return bool(response and all(response.get(key) == value for key, value in meaning.items()))
            if kind == "compositional_signal" and task_system is not None:
                response = task_system._solve_compositional_signal({"data": {"signal": signal}})
                return bool(response and all(response.get(key) == value for key, value in meaning.items()))
            if kind == "number_practice":
                return listener.numeric_system.decode_token(signal) == meaning.get("value")
            if kind == "referent_practice":
                ledger = getattr(listener, "community_lexicon", None)
                return ledger is not None and ledger.referent_for_signal(signal) == meaning.get("referent")
        except Exception:
            return False
        return False

    def _reward_grounded_dialogue(self, speaker, listener, act, understood):
        """Give small social/learning consequences to successful practice."""
        if understood:
            bonus = 0.05 if act.get("intent") == "teach" else 0.03
            speaker.own_fitness += bonus
            listener.own_fitness += bonus
            speaker.energy = min(100.0, speaker.energy + 0.05)
            listener.energy = min(100.0, listener.energy + 0.05)
            speaker.adjust_trust(listener.id, amount=0.015, channel=3)
            listener.adjust_trust(speaker.id, amount=0.025, channel=4)
            speaker.remember_interaction(listener.id, outcome=bonus, gen_index=self.generation_index)
            listener.remember_interaction(speaker.id, outcome=bonus, gen_index=self.generation_index)
            if act.get("intent") == "teach":
                speaker.teaching_attempted = True
                listener.teaching_attempted = True
        else:
            speaker.adjust_trust(listener.id, amount=-0.005, channel=4)
            listener.adjust_trust(speaker.id, amount=-0.01, channel=4)

    def run_dialogues(self, max_pairs_per_gen=30, max_turns_per_pair=2):
        if not hasattr(self, "dialogue_log"):
            self.dialogue_log = []
        self.dialogue_metrics = {
            "pairs": 0,
            "turns": 0,
            "grounded_turns": 0,
            "grounded_exchanges": 0,
            "grounded_successes": 0,
            "teaching_exchanges": 0,
            "human_token_practice_exchanges": 0,
            "free_turns": 0,
        }

        proposed_pairs = set()
        for agent in self.agents:
            partner_id = agent.choose_conversation_partner(self.agents)
            if partner_id is None:
                continue

            key = tuple(sorted((agent.id, partner_id)))
            proposed_pairs.add(key)

        if not proposed_pairs:
            return

        proposed_list = list(proposed_pairs)
        random.shuffle(proposed_list)
        chosen_pairs = proposed_list[:max_pairs_per_gen]

        id_to_agent = {a.id: a for a in self.agents}

        for a_id, b_id in chosen_pairs:
            a = id_to_agent.get(a_id)
            b = id_to_agent.get(b_id)
            if a is None or b is None:
                continue

            room_turns = []
            last_utter = None
            last_speaker_id = None
            speaker_order = [a, b] * max_turns_per_pair

            # Two compact exchanges give each participant a chance to request
            # practice/teaching and to acknowledge an interpretable message.
            # The old alternation generated unrelated utterances each turn.
            acts = [
                self._social_dialogue_act(a, b),
                self._social_dialogue_act(b, a),
            ]
            act_index = 0
            active_act = None

            for turn_index, speaker in enumerate(speaker_order):
                listener = b if speaker is a else a

                if last_utter is not None and last_speaker_id != speaker.id:
                    try:
                        # The current speaker is the recipient of the prior
                        # turn.  This was previously (and silently) delivered
                        # back to the previous speaker instead.
                        speaker.receive_message(last_speaker_id, last_utter)
                    except Exception:
                        pass

                understood = None
                if turn_index % 2 == 0:
                    active_act = acts[act_index % len(acts)]
                    act_index += 1
                    utter = active_act["signal"] if active_act is not None else None
                elif active_act is not None:
                    understood = self._interpret_grounded_dialogue_act(speaker, active_act)
                    self._reward_grounded_dialogue(listener, speaker, active_act, understood)
                    self.dialogue_metrics["grounded_exchanges"] += 1
                    self.dialogue_metrics["grounded_successes"] += int(understood)
                    self.dialogue_metrics["teaching_exchanges"] += int(active_act.get("intent") == "teach")
                    self.dialogue_metrics["human_token_practice_exchanges"] += int(
                        active_act.get("kind") == "human_token_practice"
                    )
                    # A successful acknowledgement mirrors the learnable form;
                    # on failure the learner falls back to its own expression.
                    utter = active_act["signal"] if understood else None

                if not utter:
                    try:
                        if hasattr(speaker, "produce_addressed_utterance"):
                            utter = speaker.produce_addressed_utterance(listener)
                        else:
                            utter = speaker.produce_utterance()
                    except Exception:
                        utter = None

                if not utter:
                    utter = ""

                speaker.mark_spoken_turn()
                last_utter = utter
                last_speaker_id = speaker.id

                room_turns.append({
                    "speaker_id": speaker.id,
                    "listener_id": listener.id,
                    "utterance": utter,
                    "intent": active_act.get("intent") if active_act is not None else "free",
                    "kind": active_act.get("kind") if active_act is not None else "free",
                    "meaning": active_act.get("meaning") if active_act is not None else None,
                    "understood": understood,
                })
                self.dialogue_metrics["turns"] += 1
                if active_act is not None:
                    self.dialogue_metrics["grounded_turns"] += 1
                else:
                    self.dialogue_metrics["free_turns"] += 1

            self.dialogue_log.append({
                "generation": getattr(self, "generation_index", None),
                "pair": (a_id, b_id),
                "turns": room_turns[-10:],
            })
            self.dialogue_metrics["pairs"] += 1

        if len(self.dialogue_log) > 5000:
            self.dialogue_log = self.dialogue_log[-5000:]

        dialogue_path = getattr(self, "dialogue_log_path", "dialogue_log.txt")
        with open(dialogue_path, "a") as f:
            for d in self.dialogue_log[-5:]:
                f.write(f"Gen {d['generation']} Pair {d['pair']}\n")
                for t in d["turns"]:
                    annotation = ""
                    if t.get("kind") and t.get("kind") != "free":
                        annotation = f" [{t['intent']} {t['kind']}"
                        if t.get("meaning"):
                            meaning = ",".join(f"{key}={value}" for key, value in t["meaning"].items())
                            annotation += f" meaning={meaning}"
                        if t.get("understood") is not None:
                            annotation += f" understood={t['understood']}"
                        annotation += "]"
                    f.write(f"  A{t['speaker_id']} → A{t['listener_id']}: {t['utterance']}{annotation}\n")
                f.write("\n")

    def summarize_dialogues(self, last_n=100):
        if not hasattr(self, "dialogue_log") or not self.dialogue_log:
            return "No dialogues yet."

        recent = self.dialogue_log[-last_n:]
        pairs = {}
        for rec in recent:
            key = tuple(sorted(rec["pair"]))
            pairs.setdefault(key, 0)
            pairs[key] += len(rec["turns"])

        lines = [f"Recent dialogue pairs (last {last_n} records):"]
        for (a, b), count in sorted(pairs.items(), key=lambda x: -x[1])[:15]:
            lines.append(f"  A{a}–A{b}: {count} turns")

        return "\n".join(lines)
