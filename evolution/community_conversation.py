"""A small, safe bridge between a human transcript and community language.

The bridge intentionally recognises only a controlled subset of English.  It
converts that subset into meanings that the agents already ground together,
then asks the population for a consensus expression.  It never treats an
unrecognised sentence as an instruction and never overwrites human text.
"""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import random
import re


class CommunityConversation:
    """Poll a Ryan/Community transcript and append one consensus answer."""

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

    def __init__(self, path, settle_generations=2):
        self.path = Path(path)
        self.settle_generations = max(0, int(settle_generations))
        self.pending = None
        self.last_meaning = None
        self.metrics = {
            "pending": 0,
            "answers": 0,
            "last_agreement": 0.0,
            "last_status": "idle",
            "free_answers": 0,
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

    def _ingest_human_tokens(self, coordinator, tokens):
        """Give every agent the same opportunity to encounter human words."""
        if not tokens:
            return
        memory = getattr(coordinator, "human_token_memory", None)
        if memory is None:
            coordinator.human_token_memory = Counter()
            memory = coordinator.human_token_memory
        for token in tokens:
            memory[token] += 1

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
            except Exception:
                continue

    def _free_community_reply(self, coordinator, prompt):
        """Let agents answer an unrestricted human line with a voted proposal.

        Human tokens are taught through co-occurrence with the selected
        response, rather than being mechanically echoed back at the user.
        Repeated exchanges create shared evidence without assigning an
        invented English definition behind the scenes.
        """
        human_tokens = self._human_tokens(prompt)
        self._ingest_human_tokens(coordinator, human_tokens)
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
        return answer, votes / max(1, population)

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

        if lowered in {"help", "what can you do"}:
            return (
                "Try: 'How many is 12?', 'Show r3', 'Do a2 to r3 with 4', "
                "or teach aliases with 'Teach r3 is apple' and 'Teach a2 means move'.",
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
