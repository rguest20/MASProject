"""A small, safe bridge between a human transcript and community language.

The bridge keeps structured simulation commands separate, but ordinary human
sentences are clustered into community-created, opaque interaction modes.  It
never treats unrecognised prose as an instruction or overwrites human text.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
from pathlib import Path
import random
import re

from evolution.community_world_model import CommunityWorldModel


class ConversationMemoryMixin:
    @staticmethod
    def _sentence_shape(prompt, tokens):
        """A language-neutral surface signature used by emergent modes."""
        terminal = prompt.rstrip()[-1:] if prompt.strip() else ""
        return (terminal if terminal in {"?", "!", "."} else "plain", min(6, len(tokens)))

    def _new_intent_mode(self):
        mode_id = f"mode-{self.next_intent_mode:03d}"
        self.next_intent_mode += 1
        self.intent_modes[mode_id] = {
            "tokens": Counter(),
            "transitions": defaultdict(Counter),
            "shapes": Counter(),
            "topics": Counter(),
            "uses": 0,
        }
        return mode_id

    @staticmethod
    def _decay_counter(counter, factor, minimum=0.025):
        """Apply forgetting to a Counter while keeping it bounded and sparse."""
        for key in list(counter):
            value = float(counter[key]) * factor
            if value < minimum:
                del counter[key]
            else:
                counter[key] = value

    def _decay_conversation_evidence(self, coordinator):
        """Keep recent evidence useful without making it permanent gravity."""
        self._decay_counter(self.topic_strength, 0.88)
        self._decay_counter(self.reply_feedback, 0.995)
        self._decay_counter(self.reply_ngram_feedback, 0.990)
        self._decay_counter(self.english_bigrams, 0.985)
        self._decay_counter(self.bigram_feedback, 0.995)
        self._decay_counter(self.meaning_bigrams, 0.990)
        self._decay_counter(getattr(coordinator, "human_sentence_tokens", Counter()), 0.97)
        transitions = getattr(coordinator, "human_sentence_transitions", {}) or {}
        for left in list(transitions):
            self._decay_counter(transitions[left], 0.89)
            if not transitions[left]:
                del transitions[left]
        for mode in self.intent_modes.values():
            self._decay_counter(mode["tokens"], 0.96)
            self._decay_counter(mode["shapes"], 0.97)
            self._decay_counter(mode["topics"], 0.90)
            for left in list(mode["transitions"]):
                self._decay_counter(mode["transitions"][left], 0.86)
                if not mode["transitions"][left]:
                    del mode["transitions"][left]
        self._decay_working_memory()

    def _decay_frame(self, frame, token_factor, transition_factor):
        self._decay_counter(frame["tokens"], token_factor)
        for left in list(frame["transitions"]):
            self._decay_counter(frame["transitions"][left], transition_factor)
            if not frame["transitions"][left]:
                del frame["transitions"][left]

    def _decay_working_memory(self):
        if self.working_frame is not None:
            self._decay_frame(self.working_frame, 0.93, 0.82)
        retained = deque(maxlen=self.parked_frames.maxlen)
        for frame in self.parked_frames:
            self._decay_frame(frame, 0.88, 0.74)
            if frame["tokens"]:
                retained.append(frame)
        self.parked_frames = retained

    @staticmethod
    def _make_working_frame(topic, mode):
        return {
            "topic": topic,
            "mode": mode,
            "tokens": Counter(),
            "transitions": defaultdict(Counter),
            "turns": 0,
        }

    def _park_working_frame(self):
        if self.working_frame is None or not self.working_frame["tokens"]:
            return
        previous = self.working_frame
        # Keep one parked version per topic; the most recent is the useful
        # one, and this avoids a frequently revisited subject consuming every
        # available parking slot.
        self.parked_frames = deque(
            (frame for frame in self.parked_frames if frame["topic"] != previous["topic"]),
            maxlen=self.parked_frames.maxlen,
        )
        self.parked_frames.append(previous)

    def _frame_affinity(self, agent, frame, tokens, topic):
        frame_words = set(frame["tokens"])
        overlap = len(frame_words & set(tokens)) / max(1, len(set(tokens)))
        topic_match = 1.0 if topic and frame["topic"] == topic else 0.0
        prototype = [word for word, _ in frame["tokens"].most_common(8)]
        semantic = self._semantic_affinity(agent, tokens, prototype)
        return (0.60 * overlap) + (0.30 * topic_match) + (0.10 * semantic)

    def _retrieve_parked_frame(self, coordinator, tokens, topic):
        """Restore only a frame for the same explicit subject.

        Semantic resemblance is useful for long-term links, but it is not a
        safe reason to revive a working-memory frame: it made a new subject
        such as ``banana`` inherit the old ``apple`` definition.  A fresh
        stated topic starts fresh; only a return to that exact topic resumes
        its parked local context.
        """
        if not self.parked_frames or not topic:
            return None
        for frame in list(self.parked_frames):
            if frame["topic"] == topic:
                self.parked_frames.remove(frame)
                return frame
        return None

    def _update_working_memory(self, coordinator, mode, tokens, topic):
        """Continue, park-and-switch, or retrieve the shared active frame."""
        action = "continued"
        if self.working_frame is None:
            self.working_frame = self._make_working_frame(topic, mode)
            action = "started"
        elif topic and self.working_frame["topic"] != topic:
            restored = self._retrieve_parked_frame(coordinator, tokens, topic)
            self._park_working_frame()
            self.working_frame = restored or self._make_working_frame(topic, mode)
            action = "retrieved" if restored is not None else "switched"

        frame = self.working_frame
        if topic:
            frame["topic"] = topic
        frame["mode"] = mode
        frame["tokens"].update(tokens)
        for left, right in zip(tokens, tokens[1:]):
            frame["transitions"][left][right] += 1
        frame["turns"] += 1
        self.metrics.update(
            working_topic=frame["topic"] or "",
            parked_frames=len(self.parked_frames),
            last_memory_action=action,
        )
        return frame

    @staticmethod
    def _ngrams(words, size):
        return [tuple(words[index:index + size]) for index in range(max(0, len(words) - size + 1))]

    def _recent_response_penalty(self, words):
        """Penalise recent community phrasing, especially repeated clauses."""
        if not words:
            return 0.0
        pairs = set(self._ngrams(words, 2))
        triples = set(self._ngrams(words, 3))
        penalty = 0.0
        for age, previous in enumerate(reversed(self.response_history)):
            recency = 0.78 ** age
            previous_pairs = set(self._ngrams(previous, 2))
            previous_triples = set(self._ngrams(previous, 3))
            pair_overlap = len(pairs & previous_pairs) / max(1, len(pairs))
            triple_overlap = len(triples & previous_triples) / max(1, len(triples))
            exact = 1.0 if tuple(words) == tuple(previous) else 0.0
            penalty += recency * ((0.60 * pair_overlap) + (0.55 * triple_overlap) + (0.85 * exact))
        return min(1.75, penalty)

    def _record_community_response(self, response, mode=None, topic=None):
        words = self._human_tokens(response)
        if words:
            self.response_history.append(words)
            self.last_reply = {
                "text": " ".join(words),
                "tokens": words,
                "mode": mode or self.active_intent_mode,
                "topic": topic or self.active_topic,
            }

    def _reply_feedback_score(self, words):
        """Return learned approval/rejection evidence for a candidate reply."""
        phrase_score = float(self.reply_feedback.get(" ".join(words), 0.0))
        ngrams = self._ngrams(words, 2) + self._ngrams(words, 3)
        ngram_score = sum(float(self.reply_ngram_feedback.get(gram, 0.0)) for gram in ngrams)
        return phrase_score + (ngram_score / max(1, len(ngrams)))

    def _bigram_strength(self, pair):
        """Evidence for using a two-word English phrase as one local unit."""
        observed = float(self.english_bigrams.get(pair, 0.0))
        feedback = float(self.bigram_feedback.get(pair, 0.0))
        meaning_support = min(0.25, 0.15 * float(self.meaning_bigrams.get(pair, 0.0)))
        # One occurrence is retained as evidence but is not automatically a
        # productive phrase.  Repetition or approval promotes it; rejection
        # makes it actively unattractive.
        repeated = max(0.0, observed - 1.0) / 2.0
        return max(-2.0, min(2.0, repeated + feedback + meaning_support))

    def _promoted_bigram_continuations(self, word, allowed_words):
        candidates = []
        for pair in set(self.english_bigrams) | set(self.meaning_bigrams):
            if pair[0] != word or pair[1] not in allowed_words:
                continue
            if self._bigram_strength(pair) > 0.0:
                candidates.append(pair[1])
        return candidates

    def _refresh_bigram_metrics(self):
        promoted = sum(
            self._bigram_strength(pair) > 0.10
            for pair in set(self.english_bigrams) | set(self.meaning_bigrams)
        )
        self.metrics.update(
            english_bigrams=len(self.english_bigrams),
            promoted_bigrams=promoted,
            meaning_bigrams=len(self.meaning_bigrams),
        )

    def _apply_human_feedback(self, coordinator, reward):
        """Apply an explicit human judgement to the last community sentence."""
        target = self.last_reply
        if not target:
            return "There is no recent community sentence to score.", 0.0

        reward = 1.0 if reward > 0 else -1.0
        phrase = target["text"]
        tokens = target["tokens"]
        self.reply_feedback[phrase] = max(-2.0, min(2.0, self.reply_feedback[phrase] + reward))
        for ngram in self._ngrams(tokens, 2) + self._ngrams(tokens, 3):
            self.reply_ngram_feedback[ngram] = max(
                -2.0, min(2.0, self.reply_ngram_feedback[ngram] + reward)
            )
        for pair in self._ngrams(tokens, 2):
            self.bigram_feedback[pair] = max(
                -2.0, min(2.0, self.bigram_feedback[pair] + reward)
            )
        self.world_model.apply_feedback(phrase, reward)
        world_metrics = self.world_model.metrics()
        self.metrics.update(
            world_edges=world_metrics["edges"],
            world_confirmed_edges=world_metrics["confirmed"],
        )
        self._refresh_bigram_metrics()

        # This reuses the agents' established utterance-feedback mechanism.
        # Approval consolidates a successful response; rejection makes that
        # exact proposal less attractive without deleting the human lesson
        # that may have contributed individual words to it.
        for agent in coordinator.agents:
            try:
                agent.learn_from_feedback(phrase, reward=reward, lr=0.35)
                preferences = agent.utter_bias["symbol_preferences"]
                for token in tokens:
                    current = float(preferences.get(token, 0.2))
                    if reward > 0:
                        preferences[token] = min(3.0, current + 0.08)
                    else:
                        preferences[token] = max(0.02, current * 0.72)
            except Exception:
                continue

        key = "positive_feedback" if reward > 0 else "negative_feedback"
        self.metrics[key] = int(self.metrics.get(key, 0)) + 1
        self.metrics["last_feedback"] = "+" if reward > 0 else "-"
        label = "accepted" if reward > 0 else "rejected"
        return f"Feedback {label} for the last community sentence.", 1.0

    @staticmethod
    def _semantic_affinity(agent, tokens, prototype):
        links = getattr(getattr(agent, "semantic_system", None), "links", {}) or {}
        if not tokens or not prototype:
            return 0.0
        evidence = []
        for token in tokens:
            for word in prototype:
                edge = links.get(token, {}).get(word, {})
                if isinstance(edge, dict):
                    evidence.append(abs(float(edge.get("w", 0.0))))
        return sum(evidence) / max(1, len(evidence))

    def _mode_similarity(self, agent, mode, tokens, shape):
        """One member's evidence that a sentence belongs in a known mode."""
        profile = mode["tokens"]
        lexical = sum(min(1.0, float(profile.get(token, 0.0))) for token in set(tokens))
        lexical /= max(1, len(set(tokens)))
        pairs = list(zip(tokens, tokens[1:]))
        sequential = sum(
            min(1.0, float(mode["transitions"].get(left, {}).get(right, 0.0)))
            for left, right in pairs
        ) / max(1, len(pairs))
        shape_match = 1.0 if mode["shapes"].get(shape, 0) else 0.0
        prototype = [word for word, _ in profile.most_common(8)]
        semantic = self._semantic_affinity(agent, tokens, prototype)
        return (0.55 * lexical) + (0.20 * sequential) + (0.10 * shape_match) + (0.15 * semantic)

    def _choose_intent_mode(self, coordinator, prompt, tokens):
        """Vote on a learned mode, or create one when evidence is weak.

        Mode identifiers are deliberately opaque.  The community's only
        evidence is its observed token/sequence/shape prototypes and agents'
        own semantic links; there is no ``question`` or ``greeting`` branch.
        """
        if not self.intent_modes:
            return self._new_intent_mode(), 1.0

        shape = self._sentence_shape(prompt, tokens)
        ballots = []
        scores_by_mode = defaultdict(list)
        for agent in coordinator.agents:
            scores = {
                mode_id: self._mode_similarity(agent, mode, tokens, shape)
                for mode_id, mode in self.intent_modes.items()
            }
            choice, score = max(scores.items(), key=lambda item: item[1])
            scores_by_mode[choice].append(score)
            # A member can explicitly express uncertainty rather than forcing
            # every novel sentence into its nearest existing mode.
            if score >= 0.22:
                ballots.append(choice)

        choice, votes, population = self._agreement(ballots, len(coordinator.agents))
        if choice is None:
            return self._new_intent_mode(), 0.0
        mean_score = sum(scores_by_mode[choice]) / max(1, len(scores_by_mode[choice]))
        agreement = votes / max(1, population)
        if mean_score < 0.22 or agreement < 0.50:
            return self._new_intent_mode(), agreement
        return choice, agreement

    def _topic_words(self, tokens):
        """Return content words suitable for a durable discussion attractor."""
        return [
            token for token in tokens
            if token not in self._TOPIC_STOP_WORDS
            and len(token) > 2
        ]

    def _statement_subject(self, prompt, tokens):
        """Return the stated subject of a simple human copular relation.

        This is structural parsing, not an intent label: in ``banana is
        yellow`` the first content word is the referent, whereas treating
        ``yellow`` as an equally likely topic causes the definition to lose
        its subject.  Questions retain ordinary topic selection.
        """
        if "?" in prompt:
            return None
        for index, token in enumerate(tokens):
            if token not in {"is", "are", "has", "have"}:
                continue
            for candidate in reversed(tokens[:index]):
                if candidate not in self._TOPIC_STOP_WORDS and len(candidate) > 2:
                    return candidate
        return None

    def _update_conversation_state(self, coordinator, prompt, tokens):
        self._decay_conversation_evidence(coordinator)
        intent_mode, agreement = self._choose_intent_mode(coordinator, prompt, tokens)
        mode = self.intent_modes[intent_mode]
        new_mode = mode["uses"] == 0
        shape = self._sentence_shape(prompt, tokens)
        topic_words = self._topic_words(tokens)
        mode["tokens"].update(tokens)
        mode["shapes"][shape] += 1
        mode["uses"] += 1
        for left, right in zip(tokens, tokens[1:]):
            mode["transitions"][left][right] += 1
        if topic_words:
            prior_strength = dict(self.topic_strength)
            # Repeated mention makes a topic persist, while a genuinely new
            # content word can move the conversation out of introductions.
            for word in topic_words:
                self.topic_strength[word] += 1
            current = self.active_topic
            # A newly formed mode should be able to establish a fresh subject
            # rather than inheriting the highest globally frequent word from
            # an older frame.  This is lexical novelty, not an English rule.
            stated_subject = self._statement_subject(prompt, tokens)
            candidates = [stated_subject] if stated_subject else topic_words
            if current not in topic_words:
                novel = [word for word in candidates if prior_strength.get(word, 0.0) <= 0.0]
                if novel:
                    candidates = novel
            best = max(
                enumerate(candidates),
                key=lambda item: (self.topic_strength[item[1]], item[1] == current, -item[0]),
            )[1]
            # A novel mode is evidence of a possible subject change.  This
            # prevents a frequent early token from permanently capturing the
            # attractor, while repeated turns still favour the current topic.
            if (
                current is None
                or (new_mode and current not in topic_words)
                or best == current
                or self.topic_strength[best] >= self.topic_strength.get(current, 0)
            ):
                self.active_topic = best
            mode["topics"].update(topic_words)

        self._update_working_memory(
            coordinator, intent_mode, tokens, self.active_topic
        )
        self.active_intent_mode = intent_mode
        self.metrics.update(
            mode=intent_mode,
            active_topic=self.active_topic or "",
            last_input_intent=intent_mode,
            last_mode_agreement=agreement,
        )
        return intent_mode

    def _choose_response_mode(self, coordinator, input_mode):
        """Let members vote for the mode whose evidence best fits a reply."""
        if len(self.intent_modes) == 1:
            return input_mode
        topic = self.active_topic
        ballots = []
        for agent in coordinator.agents:
            def response_score(item):
                mode_id, mode = item
                topic_fit = float(mode["topics"].get(topic, 0.0)) if topic else 0.0
                prototype = [word for word, _ in mode["tokens"].most_common(8)]
                semantic = self._semantic_affinity(agent, [topic] if topic else [], prototype)
                continuity = 0.20 if mode_id == input_mode else 0.0
                return (0.55 * topic_fit) + (0.25 * semantic) + continuity

            ballots.append(max(self.intent_modes.items(), key=response_score)[0])
        choice, _, _ = self._agreement(ballots, len(coordinator.agents))
        return choice or input_mode
