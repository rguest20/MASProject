"""
agents/cognition/task_system_v2.py
Task solving system (v2).
"""

import random
import re

from agents.agent_constants import REL_GT, REL_LT, REL_EQ


class TaskSystemConceptReasoningMixin:
    def _solve_narrative_chain(self, task):
        """
        Task:
        task_type: "narrative_chain"
        data: {"start": tok}

        Agent builds a short token chain:
            start → x → y → z
        Encourages temporal & causal structure.
        """
        data = task.get("data", {}) or {}
        start = data.get("start", "su")

        if hasattr(self, "_ensure_vec"):
            self._ensure_vec(start)

        chain = [start]
        cur = start

        for _ in range(3):
            nxt = None
            nbs = self.semantic_system._semantic_neighbors(cur, k=3)
            
            if nbs:
                nxt = random.choice(nbs)
            
            if not nxt and hasattr(self, "produce_utterance"):
                try:
                    nxt = (self.produce_utterance() or "").split()[0]
                except Exception:
                    nxt = None
            
            if not nxt:
                nxt = "su"
            chain.append(nxt)
            cur = nxt

        toks = chain[:]
        if random.random() < 0.5:
            toks.insert(0, "why")

        toks = [t for t in toks if isinstance(t, str) and t.strip()][:10]
        utter = " ".join(toks)

        try:
            self.own_fitness += 0.14
        except Exception:
            pass

        if hasattr(self, "_observe_tokens"):
            try:
                self._observe_tokens(toks, gain=0.10)
            except Exception:
                pass

        return {
            "agent_id": f"A{self.id}",
            "chain": chain,
            "utterance": utter,
        }

    def _solve_role_assignment(self, task):
        """
        Task:
        task_type: "role_assignment"
        data: {"event": tok}

        Agent chooses a 'doer' and 'receiver' for an event token.
        Encourages proto subject/object structure.
        """
        data = task.get("data", {}) or {}
        event = data.get("event", "muk")

        if hasattr(self, "_ensure_vec"):
            self._ensure_vec(event)

        # candidate "doers": identity tokens or neighbours
        doer = None
        receiver = None

        # try identity tokens first
        ids = list(getattr(self, "identity_tokens", []) or [])
        random.shuffle(ids)
        if ids:
            doer = random.choice(ids)

        # neighbours for receiver
        neigh = self.semantic_system._semantic_neighbors(event, k=4)

        for t in neigh:
            if isinstance(t, str) and t != event:
                receiver = t
                break

        if doer is None:
            doer = "agent_{}".format(self.id)
        if receiver is None:
            receiver = event

        anchor = None
        if hasattr(self, "_ensure_concept_token"):
            try:
                anchor = self._ensure_concept_token("role", tag="rel")
            except Exception:
                anchor = None

        toks = ["why"]
        if anchor:
            toks.append(anchor)
        toks.extend([doer, event, receiver])

        if hasattr(self, "produce_utterance") and random.random() < 0.4:
            try:
                toks.extend((self.produce_utterance() or "").split()[:1])
            except Exception:
                pass

        toks = [t for t in toks if isinstance(t, str) and t.strip()][:9]
        utter = " ".join(toks)

        try:
            self.own_fitness += 0.13
        except Exception:
            pass

        if hasattr(self, "_observe_tokens"):
            try:
                self._observe_tokens(toks, gain=0.09)
            except Exception:
                pass

        return {
            "agent_id": f"A{self.id}",
            "event": event,
            "doer": doer,
            "receiver": receiver,
            "utterance": utter,
        }

    def _solve_misunderstanding_detection(self, task):
        """
        Task:
        task_type: "misunderstanding_detection"
        data: {"utterance": <str>, "partner_id": int}

        Agent produces an interpretation / paraphrase.
        Encourages meta-language & repair tokens.
        """
        data = task.get("data", {}) or {}
        utt = data.get("utterance", "") or ""
        partner_id = data.get("partner_id", None)

        if hasattr(self, "_parse_utterance"):
            try:
                toks_in = self._parse_utterance(utt)
            except Exception:
                toks_in = utt.split()
        else:
            toks_in = utt.split()

        toks_in = [t for t in toks_in if isinstance(t, str) and t.strip()]

        # reuse some tokens as "evidence"
        reused = []
        for t in toks_in:
            if random.random() < 0.5:
                reused.append(t)
        reused = reused[:3]

        anchor = None
        if hasattr(self, "_ensure_concept_token"):
            try:
                anchor = self._ensure_concept_token("interpret", tag="ref")
            except Exception:
                anchor = None

        # add one neighbour to represent "different" reading
        alt = []
        if toks_in and random.random() < 0.6:
            nbs = self.semantic_system._semantic_neighbors(toks_in[0], k=2)
            if nbs:
                alt.append(nbs[0])

        out = ["why"]
        if anchor:
            out.append(anchor)
        out.extend(reused)
        out.extend(alt)

        if hasattr(self, "produce_utterance") and random.random() < 0.4:
            try:
                out.extend((self.produce_utterance() or "").split()[:1])
            except Exception:
                pass

        out = [t for t in out if isinstance(t, str) and t.strip()][:8]
        explanation = " ".join(out) if out else "why"

        try:
            self.own_fitness += 0.10
        except Exception:
            pass

        if reused and hasattr(self, "adjust_trust") and partner_id is not None:
            try:
                self.adjust_trust(partner_id, +0.015, channel=2)
            except Exception:
                pass

        if hasattr(self, "_observe_tokens"):
            try:
                self._observe_tokens(out, gain=0.08)
            except Exception:
                pass

        return {
            "agent_id": f"A{self.id}",
            "interpretation": explanation,
            "reused_tokens": reused,
        }

    def _solve_property_attribution(self, task):
        """
        Task:
        task_type: "property_attribution"
        data: {"target_token": tok}

        Agent proposes 1–2 'properties' for the target.
        Pushes adjective-like behaviour.
        """
        data = task.get("data", {}) or {}
        target = data.get("target_token", "su")

        if hasattr(self, "_ensure_vec"):
            self._ensure_vec(target)

        neighbours = self.semantic_system._semantic_neighbors(target, k=5)

        props = []
        for t in neighbours:
            if t != target and isinstance(t, str):
                props.append(t)
            if len(props) >= 2:
                break

        if not props:
            if hasattr(self, "_invent_token"):
                try:
                    props.append(self._invent_token(prefix="zev"))
                except Exception:
                    props.append("zev")
            else:
                props.append("zev")

        anchor = None
        if hasattr(self, "_ensure_concept_token"):
            try:
                anchor = self._ensure_concept_token("property", tag="ref")
            except Exception:
                anchor = None

        toks = ["why"]
        if anchor:
            toks.append(anchor)
        toks.append(target)
        toks.extend(props)

        if hasattr(self, "produce_utterance") and random.random() < 0.4:
            try:
                toks.extend((self.produce_utterance() or "").split()[:1])
            except Exception:
                pass

        toks = [t for t in toks if isinstance(t, str) and t.strip()][:9]
        utter = " ".join(toks)

        try:
            self.own_fitness += 0.11
        except Exception:
            pass

        if hasattr(self, "_observe_tokens"):
            try:
                self._observe_tokens(toks, gain=0.09)
            except Exception:
                pass

        return {
            "agent_id": f"A{self.id}",
            "target": target,
            "properties": props,
            "utterance": utter,
        }

    def _solve_verb_noun_compat(self, task):
        """
        Task:
        task_type: "verb_noun_compat"
        data: {"verb_token": tok}

        Agent proposes likely 'arguments' (nouns) for a verb-like token.
        """
        data = task.get("data", {}) or {}
        verb = data.get("verb_token", "muk")

        if hasattr(self, "_ensure_vec"):
            self._ensure_vec(verb)

        # candidate nouns from neighbours + recent tokens
        cands = []
        cands.extend(self.semantic_system._semantic_neighbors(verb, k=6))

        recent = getattr(self, "recent_tokens", [])[-15:]
        for t in recent:
            if isinstance(t, str):
                cands.append(t)

        cands = [t for t in cands if isinstance(t, str) and t != verb]
        random.shuffle(cands)
        args = list(dict.fromkeys(cands))[:3]  # unique, max 3

        if not args:
            args = ["su"]

        anchor = None
        if hasattr(self, "_ensure_concept_token"):
            try:
                anchor = self._ensure_concept_token("bind", tag="rel")
            except Exception:
                anchor = None

        toks = ["why"]
        if anchor:
            toks.append(anchor)
        toks.append(verb)
        toks.extend(args)

        toks = [t for t in toks if isinstance(t, str) and t.strip()][:9]
        utter = " ".join(toks)

        try:
            self.own_fitness += 0.11
        except Exception:
            pass

        if hasattr(self, "_observe_tokens"):
            try:
                self._observe_tokens(toks, gain=0.09)
            except Exception:
                pass

        return {
            "agent_id": f"A{self.id}",
            "verb": verb,
            "arguments": args,
            "utterance": utter,
        }

    def _solve_definition_swap(self, task):
        """
        Task:
        task_type: "definition_swap"
        data: {"token": tok, "partner_definition": <str>}

        Agent restates or tweaks a partner's definition.
        Encourages synonymy + fine-grained concept structure.
        """
        data = task.get("data", {}) or {}
        token = data.get("token", "su")
        partner_def = data.get("partner_definition", "") or ""

        if hasattr(self, "_parse_utterance"):
            try:
                p_toks = self._parse_utterance(partner_def)
            except Exception:
                p_toks = partner_def.split()
        else:
            p_toks = partner_def.split()

        p_toks = [t for t in p_toks if isinstance(t, str) and t.strip()]

        # reuse some partner tokens
        reused = []
        for t in p_toks:
            if random.random() < 0.5:
                reused.append(t)
        reused = reused[:3]

        # add one or two neighbours of token as "extra nuance"
        nuance = self.semantic_system._semantic_neighbors(token, k=2)

        anchor = None
        if hasattr(self, "_ensure_concept_token"):
            try:
                anchor = self._ensure_concept_token("define", tag="ref")
            except Exception:
                anchor = None

        toks = ["why"]
        if anchor:
            toks.append(anchor)
        toks.append(token)
        toks.extend(reused)
        toks.extend(nuance)

        toks = [t for t in toks if isinstance(t, str) and t.strip()][:10]
        utter = " ".join(toks) if toks else token

        try:
            self.own_fitness += 0.12
        except Exception:
            pass

        if hasattr(self, "_observe_tokens"):
            try:
                self._observe_tokens(toks, gain=0.10)
            except Exception:
                pass

        return {
            "agent_id": f"A{self.id}",
            "token": token,
            "utterance": utter,
            "reused_tokens": reused,
        }

    def _solve_prediction_task(self, task):
        """
        Task:
        task_type: "prediction_task"
        data: {"prefix": <utterance>}

        Agent predicts a likely next token and produces a brief 'why' chain.
        """
        data = task.get("data", {}) or {}
        prefix = data.get("prefix", "") or ""

        if hasattr(self, "_parse_utterance"):
            try:
                toks_in = self._parse_utterance(prefix)
            except Exception:
                toks_in = prefix.split()
        else:
            toks_in = prefix.split()

        toks_in = [t for t in toks_in if isinstance(t, str) and t.strip()]
        if not toks_in:
            toks_in = ["su"]

        last = toks_in[-1]

        # choose candidate via neighbours
        candidate = None
        nbs = self.semantic_system._semantic_neighbors(last, k=3)
        if nbs:
            candidate = random.choice(nbs)
        # fallback: use language organ
        if candidate is None and hasattr(self, "produce_utterance"):
            try:
                candidate = (self.produce_utterance() or "").split()[0]
            except Exception:
                candidate = None

        if candidate is None:
            candidate = "tol"

        anchor = None
        if hasattr(self, "_ensure_concept_token"):
            try:
                anchor = self._ensure_concept_token("predict", tag="rel")
            except Exception:
                anchor = None

        toks = ["why"]
        if anchor:
            toks.append(anchor)
        toks.append(last)
        toks.append(candidate)

        toks = [t for t in toks if isinstance(t, str) and t.strip()][:8]
        explanation = " ".join(toks)

        try:
            self.own_fitness += 0.10
        except Exception:
            pass

        if hasattr(self, "_observe_tokens"):
            try:
                self._observe_tokens(toks, gain=0.08)
            except Exception:
                pass

        return {
            "agent_id": f"A{self.id}",
            "prefix_last": last,
            "prediction": candidate,
            "utterance": explanation,
        }
