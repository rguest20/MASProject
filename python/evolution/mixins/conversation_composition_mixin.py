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


class ConversationCompositionMixin:
    def _ingest_human_tokens(self, coordinator, tokens, prompt=None):
        """Give every agent the same opportunity to encounter human words."""
        if not tokens:
            return
        memory = getattr(coordinator, "human_token_memory", None)
        if memory is None:
            coordinator.human_token_memory = Counter()
            memory = coordinator.human_token_memory
        origins = getattr(coordinator, "human_token_origins", None)
        if origins is None:
            coordinator.human_token_origins = {}
            origins = coordinator.human_token_origins

        dictionary_relations = {}
        meaning_evidence = {}
        dictionary = getattr(coordinator, "human_dictionary", None)
        if dictionary is not None:
            for token in tokens:
                relations = dictionary.semantic_relations(token)
                if relations is not None:
                    dictionary_relations[token] = relations
                evidence = dictionary.meaning_evidence(token)
                if evidence is not None:
                    meaning_evidence[token] = evidence
                    self.world_model.observe_dictionary(token, evidence, relations)

        if prompt:
            self.world_model.observe_human_text(prompt)

        public = set(coordinator.community_lexicon.numeric_conventions().values())
        public.update(coordinator.community_lexicon.referential_conventions().values())
        public.update(coordinator.community_lexicon.action_conventions().values())

        for token in tokens:
            # Ryan's transcript is the authoritative source for human
            # language.  An agent may already know a word from a story, but
            # that must not erase the word from the human sentence graph.
            # Only an explicitly public community signal stays outside this
            # pool; treating every private agent vocabulary item as community
            # language was the direct cause of one-word replies after reading.
            if token in origins:
                memory[token] += 1
            elif token not in public:
                origins[token] = "human"
                memory[token] += 1

        sentence_tokens = [token for token in tokens if origins.get(token) == "human"]
        sentence_pool = getattr(coordinator, "human_sentence_tokens", None)
        if sentence_pool is None:
            coordinator.human_sentence_tokens = Counter()
            sentence_pool = coordinator.human_sentence_tokens
        transitions = getattr(coordinator, "human_sentence_transitions", None)
        if transitions is None:
            from collections import defaultdict
            coordinator.human_sentence_transitions = defaultdict(Counter)
            transitions = coordinator.human_sentence_transitions
        sentence_pool.update(sentence_tokens)
        for left, right in zip(sentence_tokens, sentence_tokens[1:]):
            transitions[left][right] += 1
            self.english_bigrams[(left, right)] += 1
        for evidence in meaning_evidence.values():
            for pair in evidence["category_bigrams"]:
                # Dictionary phrases are semantic scaffolds, not direct human
                # utterances.  They need repeated exposure or feedback before
                # they can compete with observed English bigrams.
                self.meaning_bigrams[pair] += 0.35
            for word in evidence["category_tokens"]:
                sentence_pool[word] += 0.12
        self._refresh_bigram_metrics()
        self.metrics["meaning_tokens"] = len({
            word for evidence in meaning_evidence.values()
            for word in evidence["category_tokens"]
        })
        # Raw synonyms remain semantic evidence only.  Letting every lookup
        # enter the productive pool caused sense leakage such as
        # ``game -> plot`` and ``bigger -> magnanimous``.

        for agent in coordinator.agents:
            try:
                agent.vocab.update(tokens)
                agent.recent_tokens.extend(tokens)
                agent.recent_tokens = agent.recent_tokens[-64:]
                preferences = agent.utter_bias["symbol_preferences"]
                for token in tokens:
                    preferences[token] = max(preferences.get(token, 0.2), 1.0)
                    agent.semantic_system.ensure_vec(token)
                    agent._ensure_token_semantic(token)
                # Co-occurrence is the only initial signal: the community is
                # not told that an English word has a fixed meaning.
                agent._observe_language_tokens(tokens, gain=0.20)
                for token, relations in dictionary_relations.items():
                    related = relations["synonyms"] + relations["antonyms"]
                    agent.dict_vocab.update(related)
                    agent.semantic_system.ensure_vec(token)
                    for word in relations["synonyms"]:
                        agent.semantic_system.ensure_vec(word)
                        agent.semantic_system.link(token, word, +0.12)
                    for word in relations["antonyms"]:
                        agent.semantic_system.ensure_vec(word)
                        agent.semantic_system.link(token, word, -0.10)
                for token, evidence in meaning_evidence.items():
                    # Categories carry stronger, safer evidence than arbitrary
                    # definition words.  Definition links remain weak so a
                    # gloss such as "red or yellow or green" does not turn
                    # into a false equivalence for every apple.
                    agent.dict_vocab.update(evidence["category_tokens"])
                    agent.semantic_system.ensure_vec(token)
                    for word in evidence["category_tokens"]:
                        agent.semantic_system.ensure_vec(word)
                        agent.semantic_system.link(token, word, +0.08)
                    for word in evidence["definition_tokens"]:
                        agent.semantic_system.ensure_vec(word)
                        agent.semantic_system.link(token, word, +0.018)
                    for left, right in evidence["category_bigrams"]:
                        agent.semantic_system.ensure_vec(left)
                        agent.semantic_system.ensure_vec(right)
                        agent.semantic_system.link(left, right, +0.06)
            except Exception:
                continue
        world_metrics = self.world_model.metrics()
        self.metrics.update(
            world_edges=world_metrics["edges"],
            world_confirmed_edges=world_metrics["confirmed"],
        )

    def _free_community_reply(self, coordinator, prompt):
        """Let agents answer an unrestricted human line with a voted proposal.

        Human tokens are taught through co-occurrence with the selected
        response, rather than being mechanically echoed back at the user.
        Repeated exchanges create shared evidence without assigning an
        invented English definition behind the scenes.
        """
        human_tokens = self._human_tokens(prompt)
        self._update_conversation_state(coordinator, prompt, human_tokens)
        self._ingest_human_tokens(coordinator, human_tokens, prompt=prompt)
        proposals = []
        public = set(coordinator.community_lexicon.numeric_conventions().values())
        public.update(coordinator.community_lexicon.referential_conventions().values())
        public.update(coordinator.community_lexicon.action_conventions().values())

        for agent in coordinator.agents:
            try:
                act = coordinator._grounded_dialogue_act(agent, listener=None)
                phrase = act["signal"] if act is not None else agent.produce_utterance()
            except Exception:
                phrase = ""
            words = [word for word in str(phrase).split() if word]
            if not words:
                continue
            proposals.append(" ".join(words))

        if not proposals:
            return "...", 0.0

        def vote(voter, proposal):
            words = proposal.split()
            familiar = sum(word in voter.vocab for word in words) / max(1, len(words))
            grounded = sum(word in public for word in words) / max(1, len(words))
            preference = sum(
                voter.utter_bias["symbol_preferences"].get(word, 0.0)
                for word in words
            ) / max(1, len(words))
            return (0.60 * familiar) + (0.35 * grounded) + (0.05 * preference)

        ballot = []
        for voter in coordinator.agents:
            best = max(proposals, key=lambda proposal: vote(voter, proposal))
            ballot.append(best)
        answer, votes, population = self._agreement(ballot, len(coordinator.agents))
        answer = answer or random.choice(proposals)

        response_tokens = answer.split()
        for agent in coordinator.agents:
            try:
                agent._observe_language_tokens(human_tokens + response_tokens, gain=0.25)
                agent.learn_from_feedback(answer, reward=0.08, lr=0.03)
            except Exception:
                continue
        self.metrics["free_answers"] = int(self.metrics.get("free_answers", 0)) + 1
        self._record_community_response(answer, mode=self.active_intent_mode, topic=self.active_topic)
        return answer, votes / max(1, population)

    def _reading_bridge_for_context(self, coordinator, anchors):
        """Return confident reading words whose observed context is live now.

        Reading never creates a world-model assertion.  This only makes a
        word eligible for composition when its own nearby reading context
        overlaps a current human/topic word (or its dictionary category).
        """
        bridge = getattr(coordinator, "reading_conversation_tokens", {}) or {}
        contexts = getattr(coordinator, "reading_token_contexts", {}) or {}
        if not bridge or not anchors:
            self.metrics["reading_bridge_available"] = 0
            return Counter()

        # Do not expand the anchor set through dictionary categories here.
        # Broad labels such as ``thing`` or ``object`` connect unrelated
        # definitions and let a large reading vocabulary bypass the topic.
        # Equally, grammatical glue (``are``, ``you``, ``the``) cannot be a
        # relevance link: it was enough to make Peter Pan appear when Ryan
        # wrote ``You are community``.
        expanded_anchors = set(self._topic_words(anchors))

        eligible = Counter()
        for word, confidence in bridge.items():
            neighbours = contexts.get(word, {}) or {}
            contextual_evidence = sum(
                float(neighbours.get(anchor, 0.0)) for anchor in expanded_anchors
            )
            if word in expanded_anchors or contextual_evidence > 0.0:
                # Keep reading vocabulary low-weight until Ryan actually uses
                # it; it can assist a sentence but not drown out the live
                # working frame.
                eligible[word] = 0.10 + (0.20 * float(confidence)) + min(0.16, 0.04 * contextual_evidence)
        self.metrics["reading_bridge_available"] = len(eligible)
        return eligible

    def _human_sentence_reply(self, coordinator, prompt):
        """Have agents propose and vote on a sentence from human word material.

        There are no fixed English sentence templates here.  Word order comes
        from sequences Ryan has written; when a continuation is absent, an
        agent may step to a nearby dictionary word.  The result is primitive,
        but it is genuinely constructed from the community's growing human
        token graph rather than filled into a predefined phrase.
        """
        prompt_tokens = self._human_tokens(prompt)
        input_mode = self._update_conversation_state(coordinator, prompt, prompt_tokens)
        response_mode = self._choose_response_mode(coordinator, input_mode)
        self.metrics["last_reply_intent"] = response_mode
        self._ingest_human_tokens(coordinator, prompt_tokens, prompt=prompt)
        human_pool = getattr(coordinator, "human_sentence_tokens", {}) or {}
        if not human_pool:
            return None
        transitions = getattr(coordinator, "human_sentence_transitions", {}) or {}
        response_profile = self.intent_modes[response_mode]
        reply_pool = response_profile["tokens"]
        reply_transitions = response_profile["transitions"]
        dictionary = getattr(coordinator, "human_dictionary", None)
        topic = self.active_topic
        working_frame = self.working_frame or self._make_working_frame(topic, input_mode)
        working_pool = working_frame["tokens"]
        working_transitions = working_frame["transitions"]

        def world_words(word):
            return list(self.world_model.candidates(word).keys())

        allowed_words = set(working_pool)
        if topic:
            allowed_words.update(world_words(topic))
            meaning = dictionary.meaning_evidence(topic) if dictionary is not None else None
            if meaning:
                allowed_words.update(meaning["category_tokens"])
            # A graph-supported predicate is grammar needed to render one
            # subject-bound fact, not a template.  Without it ``apple`` could
            # jump straight to ``green`` and the fallback would then mix in
            # words from another definition.
            relations = self.world_model.facts(topic, minimum=0.18)
            if any(relation == "has" for relation, _, _ in relations):
                allowed_words.add("has")
            if any(relation != "has" for relation, _, _ in relations):
                allowed_words.add("is")

        bridge_pool = self._reading_bridge_for_context(
            coordinator, allowed_words | set(prompt_tokens) | ({topic} if topic else set())
        )
        pool = Counter(human_pool)
        for word, weight in bridge_pool.items():
            pool[word] = max(float(pool.get(word, 0.0)), weight)
        allowed_words.update(bridge_pool)
        reading_transitions = getattr(coordinator, "reading_sentence_transitions", {}) or {}
        foreign_subjects = set()
        for key in getattr(self.world_model, "edges", {}):
            _, subject, _ = key
            human_evidence = getattr(self.world_model, "sources", {}).get(key, {})
            if subject != topic and float(human_evidence.get("human_assertion", 0.0)) > 0.0:
                foreign_subjects.add(subject)

        def subject_safe(word):
            # A subject from another learned definition should never be a
            # fallback completion for the current subject.  It can still be
            # discussed after Ryan makes it the active topic.
            return word not in foreign_subjects

        def choose_word(agent, candidates, previous=None):
            candidates = list(dict.fromkeys(
                word for word in candidates
                if word in pool and word in allowed_words and subject_safe(word)
            ))
            if not candidates:
                return None
            prefs = agent.utter_bias["symbol_preferences"]
            weights = []
            for word in candidates:
                phrase_evidence = self._bigram_strength((previous, word)) if previous else 0.0
                phrase_factor = (
                    1.0 + (0.75 * max(0.0, phrase_evidence))
                    if phrase_evidence >= 0.0
                    else max(0.03, 1.0 + phrase_evidence)
                )
                weights.append(
                    max(0.05, float(working_pool.get(word, 0.05)))
                    * (0.5 + float(prefs.get(word, 0.2)))
                    * (1.0 + 0.35 * float(reply_pool.get(word, 0.0)))
                    * (1.0 + (0.20 * float(bridge_pool.get(word, 0.0))))
                    * (2.5 if word == topic else 1.0)
                    * phrase_factor
                )
            return random.choices(candidates, weights=weights, k=1)[0]

        proposals = []
        topic_starts = [topic] if topic in pool else []
        frame_starts = list(working_pool)
        prompt_starts = [word for word in prompt_tokens if word in allowed_words]
        starts = topic_starts * 3 + frame_starts + prompt_starts
        for agent in coordinator.agents:
            # Once the community has a live topic, every proposal must carry
            # it.  Without this anchor an old global transition can begin at
            # an unrelated word and recreate a stale phrase before topical
            # evidence has any chance to affect the vote.
            start = topic if topic in pool else choose_word(agent, starts)
            if start is None:
                continue
            words = [start]
            target_length = random.randint(3, 6)
            while len(words) < target_length:
                # Stop once this proposal has expressed one complete,
                # subject-bound graph fact.  Continuing from its object was
                # the main source of tails such as ``apple is fruit an
                # green`` and let unrelated working-memory words leak in.
                if (
                    len(words) >= 3
                    and words[0] == topic
                    and self.world_model.supports_clause(words[0], words[1], words[2])
                ):
                    break
                current = words[-1]
                continuation = self._promoted_bigram_continuations(current, allowed_words)
                continuation += list((working_transitions.get(current, {}) or {}).keys())
                continuation += self.world_model.continuations(
                    current, allowed_words, subject=topic
                )
                # Reading order is a private syntactic hint, not a human
                # observation and not a fact in the community world model.
                # It can only lead to another context-relevant bridge word.
                continuation += [
                    word for word in (reading_transitions.get(current, {}) or {})
                    if word in bridge_pool
                ]
                if not continuation:
                    continuation = list((reply_transitions.get(current, {}) or {}).keys())
                if not continuation:
                    continuation = [
                        word for word in (transitions.get(current, {}) or {})
                        if word in allowed_words
                    ]
                if not continuation:
                    continuation = list(working_pool)
                unseen = [word for word in continuation if word not in words]
                if unseen:
                    continuation = unseen
                else:
                    remaining = [
                        word for word in pool
                        if word not in words and subject_safe(word)
                    ]
                    if not remaining:
                        break
                    continuation = remaining
                next_word = choose_word(agent, continuation, previous=current)
                if next_word is None:
                    break
                words.append(next_word)
            # A lone topic word is an activation trace, not a conversational
            # answer.  Keep it in working/long-term learning, but require at
            # least a small phrase before members can put it in converse.txt.
            if len(words) >= 3:
                proposals.append(" ".join(words) + ".")

        if not proposals:
            # The community should expose uncertainty rather than presenting
            # one word as though it were a complete thought.  The topic token
            # remains in memory and the next human example supplies more
            # compositional evidence.
            request = (
                f"We need more context for {topic}."
                if topic else "We need more context before we can answer."
            )
            self._record_community_response(request, mode=response_mode, topic=topic)
            return request, 1.0

        prompt_phrase = " ".join(prompt_tokens)

        def sentence_score(voter, sentence):
            words = self._human_tokens(sentence)
            if not words:
                return -1.0
            observed_pairs = sum(
                float((transitions.get(left, {}) or {}).get(right, 0))
                for left, right in zip(words, words[1:])
            )
            working_pairs = sum(
                float((working_transitions.get(left, {}) or {}).get(right, 0))
                for left, right in zip(words, words[1:])
            )
            intent_pairs = sum(
                float((reply_transitions.get(left, {}) or {}).get(right, 0))
                for left, right in zip(words, words[1:])
            )
            bigram_evidence = sum(
                self._bigram_strength((left, right))
                for left, right in zip(words, words[1:])
            ) / max(1, len(words) - 1)
            world_evidence = self.world_model.relation_score(words)
            semantic_pairs = 0.0
            for left, right in zip(words, words[1:]):
                edge = getattr(voter.semantic_system, "links", {}).get(left, {}).get(right, {})
                semantic_pairs += abs(float(edge.get("w", 0.0))) if isinstance(edge, dict) else 0.0
            familiar = sum(word in voter.vocab or word in voter.dict_vocab for word in words) / len(words)
            frame_fit = sum(float(working_pool.get(word, 0.0)) for word in words) / len(words)
            topical = sum(word == topic for word in words) / len(words) if topic else 0.0
            if topic:
                topical += sum(
                    abs(float(getattr(voter.semantic_system, "links", {}).get(topic, {}).get(word, {}).get("w", 0.0)))
                    for word in words if word != topic
                ) / len(words)
            copied_prompt = 1.0 if " ".join(words) == prompt_phrase else 0.0
            repetition = self._recent_response_penalty(words)
            feedback = self._reply_feedback_score(words)
            feedback_effect = (
                (0.12 * max(0.0, feedback))
                - (1.20 * max(0.0, -feedback))
            )
            bigram_effect = (
                (0.28 * max(0.0, bigram_evidence))
                - (1.20 * max(0.0, -bigram_evidence))
            )
            return (
                (0.25 * familiar) + (0.05 * observed_pairs) + (0.10 * intent_pairs)
                + (0.34 * working_pairs) + (0.14 * semantic_pairs)
                + (0.22 * frame_fit) + (0.38 * topical) + (0.30 * world_evidence)
                - (0.25 * copied_prompt) - repetition + feedback_effect + bigram_effect
            )

        ballot = [max(proposals, key=lambda sentence: sentence_score(agent, sentence)) for agent in coordinator.agents]
        sentence, votes, population = self._agreement(ballot, len(coordinator.agents))
        if sentence is None:
            return None
        response_tokens = self._human_tokens(sentence)
        self.metrics["last_repeat_penalty"] = self._recent_response_penalty(response_tokens)
        self._record_community_response(sentence, mode=response_mode, topic=topic)
        for agent in coordinator.agents:
            try:
                agent.vocab.update(response_tokens)
                agent._observe_language_tokens(prompt_tokens + response_tokens, gain=0.30)
            except Exception:
                continue
        return sentence, votes / max(1, population)
