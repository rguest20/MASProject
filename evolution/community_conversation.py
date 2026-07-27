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


class CommunityConversation:
    """Poll a Ryan/Community transcript and append one consensus answer."""

    # This is only a small structural-word filter for topic salience.  It is
    # deliberately not a catalogue of English speech acts or intent labels.
    _TOPIC_STOP_WORDS = frozenset({
        "a", "an", "about", "and", "are", "be", "can", "do", "does", "for",
        "have", "how", "i", "im", "i'm", "is", "it", "it's", "me", "my", "of",
        "on", "tell", "the", "to", "we", "what", "when", "where", "which", "who",
        "why", "with", "you", "your",
    })

    _PREFIX = re.compile(r"^\s*(Ryan|Community)\s*:\s*(.*)$", re.IGNORECASE)
    _TEACH_ENTITY = re.compile(
        r"^teach\s+(r[0-5]|a[0-3])\s+(?:is|means)\s+([a-z][a-z-]{1,31})\s*[.!?]*$",
        re.IGNORECASE,
    )
    _TEACH_NUMBER = re.compile(
        r"^teach\s+(?:number\s+)?(\d+)\s+(?:is|means)\s+([a-z][a-z-]{1,31})\s*[.!?]*$",
        re.IGNORECASE,
    )
    _NUMBER = re.compile(r"^(?:how many is|what is)\s+(\d+)\s*[?!.]*$", re.IGNORECASE)
    _SHOW = re.compile(r"^(?:show|name)\s+([a-z0-9-]+)\s*[?!.]*$", re.IGNORECASE)
    _DO = re.compile(
        r"^(?:do|please do)\s+([a-z0-9-]+)\s+(?:to|on)\s+([a-z0-9-]+)"
        r"(?:\s+(?:with|using)\s+([a-z0-9-]+))?\s*[?!.]*$",
        re.IGNORECASE,
    )
    _FEEDBACK = re.compile(r"^(?:feedback\s*)?([+-])\s*$", re.IGNORECASE)
    # These are graph-shaped questions, rather than a catalogue of named
    # intents.  They are handled before free composition so the community can
    # retrieve one learned triple instead of sampling from every known word.
    _WORLD_SUBJECT_QUERY = re.compile(
        r"^(?:what|who)\s+(is|are|has|have)\s+(?:(?:the|an|a)\s+)?"
        r"([a-z][a-z'-]{1,31})\s*[?!.]*$",
        re.IGNORECASE,
    )
    _WORLD_REVERSE_QUERY = re.compile(
        r"^(?:what|who)\s+(has|have)\s+(?:(?:the|an|a)\s+)?"
        r"([a-z][a-z'-]{1,31})\s*[?!.]*$",
        re.IGNORECASE,
    )

    def __init__(self, path, settle_generations=2):
        self.path = Path(path)
        self.settle_generations = max(0, int(settle_generations))
        self.pending = None
        self.last_meaning = None
        self.active_intent_mode = None
        self.intent_modes = {}
        self.next_intent_mode = 1
        self.active_topic = None
        self.topic_strength = Counter()
        # A shared scratchpad is intentionally small.  Durable agent semantic
        # links keep learning alive; this structure controls only what is
        # currently eligible to steer the next community reply.
        self.working_frame = None
        self.parked_frames = deque(maxlen=4)
        # Community output is not added to the human sentence graph, but a
        # short history lets members notice when they are merely repeating it.
        self.response_history = deque(maxlen=8)
        self.last_reply = None
        self.reply_feedback = Counter()
        self.reply_ngram_feedback = Counter()
        # English bigrams are phrase candidates, distinct from raw next-word
        # counts.  They become productive only with repeated human evidence
        # or explicit positive feedback.
        self.english_bigrams = Counter()
        self.bigram_feedback = Counter()
        self.meaning_bigrams = Counter()
        self.world_model = CommunityWorldModel()
        self.metrics = {
            "pending": 0,
            "answers": 0,
            "last_agreement": 0.0,
            "last_status": "idle",
            "free_answers": 0,
            "mode": "",
            "active_topic": "",
            "last_input_intent": "",
            "last_reply_intent": "",
            "last_mode_agreement": 0.0,
            "last_repeat_penalty": 0.0,
            "working_topic": "",
            "parked_frames": 0,
            "last_memory_action": "idle",
            "positive_feedback": 0,
            "negative_feedback": 0,
            "last_feedback": "",
            "english_bigrams": 0,
            "promoted_bigrams": 0,
            "meaning_tokens": 0,
            "meaning_bigrams": 0,
            "world_edges": 0,
            "world_confirmed_edges": 0,
        }

    @staticmethod
    def _signature(line_number, text):
        raw = f"{line_number}:{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _transcript(self):
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            return None

    def _latest_unanswered_prompt(self, text):
        """Return the most recent complete Ryan line without a reply."""
        entries = []
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            match = self._PREFIX.match(raw_line)
            if not match:
                continue
            role, body = match.groups()
            body = body.strip()
            if body:
                entries.append((line_number, role.lower(), body))

        latest = None
        for index, (line_number, role, body) in enumerate(entries):
            if role != "ryan":
                continue
            next_role = entries[index + 1][1] if index + 1 < len(entries) else None
            if next_role != "community":
                latest = (line_number, body)
        return latest

    def _aliases_from_transcript(self, text):
        aliases = {}
        for raw_line in text.splitlines():
            match = self._PREFIX.match(raw_line)
            if not match or match.group(1).lower() != "ryan":
                continue
            body = match.group(2).strip()
            entity = self._TEACH_ENTITY.match(body)
            if entity:
                key, word = entity.groups()
                aliases[word.lower()] = key.lower()
                continue
            number = self._TEACH_NUMBER.match(body)
            if number:
                value, word = number.groups()
                aliases[word.lower()] = int(value)
        return aliases

    @staticmethod
    def _agreement(values, population_size):
        values = [value for value in values if value]
        if not values:
            return None, 0, population_size
        choice, count = Counter(values).most_common(1)[0]
        return choice, count, population_size

    @staticmethod
    def _resolve(value, aliases, prefix=None):
        value = (value or "").strip().lower()
        resolved = aliases.get(value, value)
        if prefix and isinstance(resolved, str) and not resolved.startswith(prefix):
            return None
        return resolved

    @staticmethod
    def _resolve_number(value, aliases):
        value = (value or "").strip().lower()
        resolved = aliases.get(value, value)
        if isinstance(resolved, int):
            return resolved
        try:
            return int(resolved)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _answer_text(body):
        return f"Community: {body}\nRyan: \n"

    @staticmethod
    def _human_tokens(prompt):
        """Keep only bounded word-like tokens; transcript text is never executed."""
        return re.findall(r"[a-z][a-z'-]{0,31}", prompt.lower())[:12]

    def _world_question_reply(self, coordinator, prompt):
        """Answer a simple graph question from one subject-bound fact set.

        A question is still observed by the community's normal intent and
        working-memory machinery.  Only the final retrieval is constrained:
        the answer can use a fact about the requested subject (or, for a
        possessive reverse query, a subject that owns the requested object).
        """
        cleaned = prompt.strip()
        reverse = self._WORLD_REVERSE_QUERY.match(cleaned)
        direct = self._WORLD_SUBJECT_QUERY.match(cleaned)
        if reverse is None and direct is None:
            return None

        tokens = self._human_tokens(cleaned)
        self._update_conversation_state(coordinator, cleaned, tokens)
        self._ingest_human_tokens(coordinator, tokens, prompt=cleaned)

        if reverse is not None:
            verb, obj = reverse.groups()
            subjects = self.world_model.subjects(verb, obj, minimum=0.50)
            if subjects:
                subject, relation, _ = subjects[0]
                answer = self.world_model.render_fact(subject, relation, obj)
            else:
                answer = f"I do not yet have a confirmed fact involving {obj.lower()}."
        else:
            verb, subject = direct.groups()
            facts = self.world_model.facts(subject, relation=verb, minimum=0.50)
            if facts:
                relation, obj, _ = facts[0]
                answer = self.world_model.render_fact(subject.lower(), relation, obj)
            else:
                answer = f"I do not yet have a confirmed fact about {subject.lower()}."

        self._record_community_response(
            answer,
            mode=self.active_intent_mode,
            topic=self.active_topic,
        )
        return answer, 1.0

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
        """Return a parked frame only when members recognise a connection."""
        if not self.parked_frames:
            return None
        ballots = []
        for agent in coordinator.agents:
            best = max(
                self.parked_frames,
                key=lambda frame: self._frame_affinity(agent, frame, tokens, topic),
            )
            if self._frame_affinity(agent, best, tokens, topic) >= 0.30:
                ballots.append(best["topic"])
        selected_topic, votes, population = self._agreement(ballots, len(coordinator.agents))
        if selected_topic is None or votes / max(1, population) < 0.50:
            return None
        for frame in list(self.parked_frames):
            if frame["topic"] == selected_topic:
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
            candidates = topic_words
            if current not in topic_words:
                novel = [word for word in topic_words if prior_strength.get(word, 0.0) <= 0.0]
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
        known_to_community = set(public)
        for agent in coordinator.agents:
            known_to_community.update(getattr(agent, "vocab", set()))

        for token in tokens:
            # A human quoting ``belmuk`` is useful evidence about how they
            # attend to community language, but it is not an English word the
            # agents need to rehearse as a new discourse item.
            if token in origins:
                memory[token] += 1
            elif token not in known_to_community:
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
        pool = getattr(coordinator, "human_sentence_tokens", {}) or {}
        if not pool:
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

        def choose_word(agent, candidates, previous=None):
            candidates = list(dict.fromkeys(
                word for word in candidates if word in pool and word in allowed_words
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
                    remaining = [word for word in pool if word not in words]
                    if not remaining:
                        break
                    continuation = remaining
                next_word = choose_word(agent, continuation, previous=current)
                if next_word is None:
                    break
                words.append(next_word)
            proposals.append(" ".join(words) + ".")

        if not proposals:
            return None

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

    def _community_number(self, coordinator, value):
        ledger = coordinator.community_lexicon
        base = ledger.community_base()
        if base is None:
            return "We are still agreeing on a counting base; please ask again shortly.", 0.0
        proposals = []
        for agent in coordinator.agents:
            try:
                proposals.append(agent.numeric_system.speak_number(value, base=base))
            except Exception:
                pass
        phrase, votes, population = self._agreement(proposals, len(coordinator.agents))
        if phrase is None or votes * 2 <= population:
            return "We do not yet have a community answer for that quantity.", 0.0
        return f"{value} is '{phrase}' in our shared base-{base} system. Agreement: {votes}/{population}.", votes / population

    def _community_referent(self, coordinator, referent):
        token = coordinator.community_lexicon.referential_signal(referent)
        if token is None:
            return f"We have not yet agreed a public word for {referent}.", 0.0
        return f"{referent} is '{token}'. Agreement: {len(coordinator.agents)}/{len(coordinator.agents)}.", 1.0

    def _community_action(self, coordinator, action, referent, value):
        ledger = coordinator.community_lexicon
        action_token = ledger.action_signal(action)
        referent_token = ledger.referential_signal(referent)
        number_token = ledger.numeric_token(value)
        order = ledger.grammar_order("referent_action_number")
        if not all((action_token, referent_token, number_token, order)):
            return "We need more practice before we can express that action reliably.", 0.0

        values = {"referent": referent_token, "action": action_token, "number": number_token}
        signal = " ".join(values[role] for role in order)
        act = {
            "kind": "compositional_action_signal",
            "signal": signal,
            "meaning": {"referent": referent, "action": action, "value": value},
        }
        understood = sum(
            bool(coordinator._interpret_grounded_dialogue_act(agent, act))
            for agent in coordinator.agents
        )
        if understood * 2 <= len(coordinator.agents):
            return "Our members did not reach a reliable interpretation of that action.", understood / max(1, len(coordinator.agents))
        self.last_meaning = act["meaning"]
        return (
            f"I propose '{signal}' for {action} to {referent} with {value}. "
            f"Agreement: {understood}/{len(coordinator.agents)}.",
            understood / len(coordinator.agents),
        )

    def _answer(self, coordinator, prompt, aliases):
        cleaned = prompt.strip()
        lowered = cleaned.lower().rstrip("?!.").strip()
        population = len(coordinator.agents)

        feedback = self._FEEDBACK.match(cleaned)
        if feedback:
            return self._apply_human_feedback(
                coordinator, 1.0 if feedback.group(1) == "+" else -1.0
            )

        if lowered in {"help", "what can you do"}:
            return (
                "Try: 'How many is 12?', 'Show r3', 'Do a2 to r3 with 4', "
                "teach aliases with 'Teach r3 is apple', or score the last reply with '+' or '-'.",
                1.0,
            )
        if lowered in {"what did you do", "what did you do?"}:
            if not self.last_meaning:
                return "I have not completed a community action in this session yet.", 1.0
            meaning = self.last_meaning
            return (
                f"Our last agreed action was {meaning['action']} to {meaning['referent']} "
                f"with {meaning['value']}.",
                1.0,
            )

        taught_entity = self._TEACH_ENTITY.match(cleaned)
        if taught_entity:
            entity, word = taught_entity.groups()
            return f"Learned English bridge: {word.lower()} means {entity.lower()}.", 1.0
        taught_number = self._TEACH_NUMBER.match(cleaned)
        if taught_number:
            value, word = taught_number.groups()
            return f"Learned English bridge: {word.lower()} means {int(value)}.", 1.0

        number = self._NUMBER.match(cleaned)
        if number:
            return self._community_number(coordinator, int(number.group(1)))

        show = self._SHOW.match(cleaned)
        if show:
            referent = self._resolve(show.group(1), aliases, prefix="r")
            if referent is None:
                return "I know public referents as r0 through r5, or an alias you have taught me.", 0.0
            return self._community_referent(coordinator, referent)

        action = self._DO.match(cleaned)
        if action:
            action_id = self._resolve(action.group(1), aliases, prefix="a")
            referent = self._resolve(action.group(2), aliases, prefix="r")
            value = self._resolve_number(action.group(3) or "0", aliases)
            if action_id is None or referent is None or value is None:
                return "Use an action a0-a3, a referent r0-r5, and a known quantity (or teach aliases first).", 0.0
            return self._community_action(coordinator, action_id, referent, value)

        world_answer = self._world_question_reply(coordinator, cleaned)
        if world_answer is not None:
            return world_answer

        human_sentence = self._human_sentence_reply(coordinator, cleaned)
        if human_sentence is not None:
            return human_sentence
        return self._free_community_reply(coordinator, cleaned)

    def poll(self, coordinator):
        """Observe one stable human prompt and append one community response."""
        text = self._transcript()
        if text is None:
            self.metrics.update(pending=0, last_status="missing")
            return None

        prompt = self._latest_unanswered_prompt(text)
        if prompt is None:
            self.pending = None
            self.metrics.update(pending=0, last_status="idle")
            return None

        line_number, body = prompt
        signature = self._signature(line_number, body)
        generation = int(getattr(coordinator, "generation_index", 0))
        if self.pending is None or self.pending["signature"] != signature:
            self.pending = {"signature": signature, "text": text, "first_seen": generation}
            self.metrics.update(pending=1, last_status="settling")
            return None
        if generation - self.pending["first_seen"] < self.settle_generations:
            self.metrics.update(pending=1, last_status="settling")
            return None

        # Do not append if a user edited the file while the community was
        # thinking.  The next generation will treat the revised text anew.
        current_text = self._transcript()
        if current_text != self.pending["text"]:
            self.pending = None
            self.metrics.update(pending=1, last_status="changed")
            return None

        aliases = self._aliases_from_transcript(current_text)
        answer, agreement = self._answer(coordinator, body, aliases)
        try:
            with self.path.open("a", encoding="utf-8") as transcript:
                if current_text and not current_text.endswith("\n"):
                    transcript.write("\n")
                transcript.write(self._answer_text(answer))
        except OSError:
            self.metrics.update(pending=1, last_status="write_failed")
            return None

        self.pending = None
        self.metrics.update(
            pending=0,
            answers=int(self.metrics.get("answers", 0)) + 1,
            last_agreement=float(agreement),
            last_status="answered",
        )
        return answer
