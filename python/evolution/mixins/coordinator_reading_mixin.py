import math
import random
import re
from collections import Counter

from agents.cognition.semantic_utils import add, scale, cos_sim
from evolution.coordinator_settings import (
    REFERENTIAL_BONUS,
    PHASE2_LR,
    REWARD_EPS,
    REWARD_TEMP,
    TEACH_PAIRS_PER_PASS,
    TEACH_ACC_TEMP,
)


class CoordinatorReadingMixin:
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

    def _refresh_reading_bridge(self, candidates=None):
        """Promote well-contextualised reading words into a private dialogue lexicon.

        Repetition alone is deliberately insufficient: common function words
        and a repeatedly quoted name would otherwise look understood.  A word
        needs independent sentence exposure and several local neighbours.  A
        dictionary entry adds a little corroboration, but never promotes a
        word by itself.  The resulting lexicon remains separate from both the
        world model and Ryan's observed sentence graph.
        """
        occurrences = getattr(self, "reading_token_memory", {}) or {}
        sentence_counts = getattr(self, "reading_token_sentences", {}) or {}
        context_keys = getattr(self, "reading_token_context_keys", {}) or {}
        contexts = getattr(self, "reading_token_contexts", {}) or {}
        bridge = getattr(self, "reading_conversation_tokens", None)
        if bridge is None:
            bridge = Counter()
            self.reading_conversation_tokens = bridge
        scores = getattr(self, "reading_bridge_scores", None)
        if scores is None:
            scores = {}
            self.reading_bridge_scores = scores

        dictionary = getattr(self, "human_dictionary", None)
        promoted = []
        candidates = occurrences if candidates is None else candidates
        for token in candidates:
            if token not in occurrences:
                continue
            if (
                len(token) < 3
                or not token.isalpha()
                or token in self._READING_BRIDGE_STOP_WORDS
            ):
                continue
            # Re-reading a refrain may consolidate private learning, but it
            # cannot masquerade as independent contextual understanding.
            sentence_count = len(context_keys.get(token, ()))
            if not sentence_count:
                sentence_count = int(sentence_counts.get(token, 0))
            neighbours = {
                word for word, evidence in (contexts.get(token, {}) or {}).items()
                if evidence > 0 and word not in self._READING_BRIDGE_STOP_WORDS
            }
            neighbour_evidence = sum(
                1 for evidence in (contexts.get(token, {}) or {}).values()
                if evidence >= 2
            )
            dictionary_support = 0.0
            if dictionary is not None:
                try:
                    dictionary_support = 1.0 if dictionary.meaning_evidence(token) else 0.0
                except Exception:
                    dictionary_support = 0.0
            score = (
                0.44 * min(1.0, sentence_count / 5.0)
                + 0.30 * min(1.0, len(neighbours) / 4.0)
                + 0.16 * min(1.0, neighbour_evidence / 3.0)
                + 0.10 * dictionary_support
            )
            scores[token] = score
            # Four independent sentences and several local neighbours are a
            # minimum for a word to leave reading quarantine.  This is slow
            # by design: a short story run should not make most of its corpus
            # immediately available as a competing conversation vocabulary.
            corroborated = dictionary_support > 0.0 or neighbour_evidence >= 2
            if (
                sentence_count >= 4
                and len(neighbours) >= 3
                and corroborated
                and score >= 0.72
                and token not in bridge
            ):
                bridge[token] = score
                promoted.append(token)
            elif token in bridge:
                bridge[token] = score
        return promoted

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
            # Count independent sentence exposure separately from raw token
            # frequency.  The latter is useful memory; the former is better
            # evidence that a word has a stable role across contexts.
            self.reading_token_sentences.update(set(tokens))
            sentence_key = tuple(tokens)
            for token in set(tokens):
                self.reading_token_context_keys[token].add(sentence_key)
            for token in tokens:
                if token not in self.reading_token_memory:
                    new_tokens.add(token)
                self.reading_token_memory[token] += 1
                self.reading_sentence_tokens[token] += 1
            for index, token in enumerate(tokens):
                # A three-word radius captures compact relations such as
                # ``king ... castle`` without treating an entire sentence as
                # one indiscriminate bag of words.
                for neighbour_index in range(max(0, index - 3), min(len(tokens), index + 4)):
                    if neighbour_index != index:
                        self.reading_token_contexts[token][tokens[neighbour_index]] += 1
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

        promoted = self._refresh_reading_bridge({
            token for tokens in tokenised for token in tokens
        })

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
            reading_bridge_promotions=len(promoted),
            reading_bridge_vocabulary=len(self.reading_conversation_tokens),
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
                    f"link-agreement={link_agreement:.3f} "
                    f"bridge=+{len(promoted)}/{len(self.reading_conversation_tokens)}\n"
                )
                if promoted:
                    log.write(f"  Conversation bridge: {', '.join(promoted)}\n")
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
            "reading_bridge_promotions",
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
