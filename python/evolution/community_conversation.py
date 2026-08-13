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

from evolution.mixins.conversation_transcript_mixin import ConversationTranscriptMixin
from evolution.mixins.conversation_memory_mixin import ConversationMemoryMixin
from evolution.mixins.conversation_composition_mixin import ConversationCompositionMixin

class CommunityConversation(ConversationTranscriptMixin, ConversationMemoryMixin, ConversationCompositionMixin):
    """Poll a Ryan/Community transcript and append one consensus answer."""

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
            "reading_bridge_available": 0,
        }

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
