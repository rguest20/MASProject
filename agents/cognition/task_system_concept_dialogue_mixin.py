"""
agents/cognition/task_system_v2.py
Task solving system (v2).
"""

import random
import re

from agents.agent_constants import REL_GT, REL_LT, REL_EQ


class TaskSystemConceptDialogueMixin:
    def _attempt_task(self, task):
        tid = task.get("task_id")
        if tid is None:
            return
        self.tasks_attempted_this_gen.add(tid)

        attempted_by = task.setdefault("attempted_by", [])
        if self.id not in attempted_by:
            attempted_by.append(self.id)

        # already solved previously? skip
        if tid in self.solved_tasks:
            return

        ttype = task.get("task_type")
        handler_name = self.TASK_HANDLERS.get(ttype)
        if handler_name is None:
            return

        try:
            resp = getattr(self, handler_name)(task)
        except Exception as e:
            # Hard shield: a bad task handler should not kill the agent loop
            print(f"[ERROR] A{self.id} in task '{ttype}': {type(e).__name__}: {e}")
            return

        # record and semantically reinforce justification tokens
        if resp:
            self.solved_tasks[tid] = resp
            responses = task.get("responses")
            if not isinstance(responses, list):
                responses = []
                task["responses"] = responses
            if not any(r.get("agent_id") == resp.get("agent_id") for r in responses):
                responses.append(resp)
            if hasattr(self, "_observe_tokens"):
                toks = resp.get("justification", "") or resp.get("utterance", "") or ""
                toks = toks.split()
                if toks:
                    try:
                        self._observe_tokens(toks, gain=0.2)
                    except Exception:
                        pass

    def _solve_token_compression(self, task):
        """
        Token-Compression / Expansion task.
        If the input utterance is long: compress it.
        If short: expand it.
        """

        if not task:
            return None

        my_id = getattr(self, "id", None)
        assigned = task.get("assigned_agents", [])
        if my_id not in assigned:
            return None

        utt = task.get("utterance", "") or ""
        toks = utt.split()

        # Determine mode
        mode = "compress" if len(toks) > 6 else "expand"

        # Compression: pick salient roots
        if mode == "compress":
            roots = []
            sem = getattr(self, "semantic", {})
            vecs = sem.get("vecs", {})
            for t in toks:
                if t in vecs and random.random() < 0.25:
                    roots.append(t)
            if not roots:
                roots = toks[:2]

            summary = " ".join(roots[:4])

            try:
                self.own_fitness += 0.15
            except Exception:
                pass

            if hasattr(self, "_observe_tokens"):
                try:
                    self._observe_tokens(roots, gain=0.05)
                except Exception:
                    pass

            return {
                "agent_id": f"A{self.id}",
                "mode": "compress",
                "result": summary,
            }

        # Expansion: add relations + neighbours
        anchor = None
        if hasattr(self, "_ensure_concept_token"):
            try:
                anchor = self._ensure_concept_token("rel", tag="rel") or "rel"
            except Exception:
                anchor = "rel"

        expanded = ["why"]
        if anchor:
            expanded.append(anchor)

        for t in toks:
            try:
                nbs = self.semantic_system._semantic_neighbors(t, k=1)
                if nbs:
                    expanded.append(nbs[0])
            except Exception:
                pass

        if hasattr(self, "produce_utterance") and random.random() < 0.7:
            try:
                expanded.extend((self.produce_utterance() or "").split()[:2])
            except Exception:
                pass

        expanded = [x for x in expanded if isinstance(x, str) and x.strip()][:8]
        sentence = " ".join(expanded)

        try:
            self.own_fitness += 0.15
        except Exception:
            pass

        if hasattr(self, "_observe_tokens"):
            try:
                self._observe_tokens(expanded, gain=0.05)
            except Exception:
                pass

        return {
            "agent_id": f"A{self.id}",
            "mode": "expand",
            "result": sentence,
        }

    def _solve_preference_alignment_dialogue(self, task):
        """
        Preference alignment:
        topic: {"A": tok1, "B": tok2, "pivot": tok3}
        Agents pick which is 'closer' to the pivot token, but
        the agreement matters more than correctness.
        """

        if not task:
            return None

        my_id = getattr(self, "id", None)
        assigned = task.get("assigned_agents", [])
        if my_id not in assigned:
            return None

        A = task.get("A", "su")
        B = task.get("B", "tol")
        P = task.get("pivot", "muk")

        # compute similarity scores if possible
        simA = 0.0
        simB = 0.0
        if hasattr(self, "semantic_distance"):
            try:
                dA = self.semantic_distance(A, P)
                dB = self.semantic_distance(B, P)
                # treat smaller distance as higher similarity
                simA = -dA
                simB = -dB
            except Exception:
                pass

        choice = A if simA >= simB else B

        # produce a reasoned utterance
        anchor = None
        if hasattr(self, "_ensure_concept_token"):
            try:
                anchor = self._ensure_concept_token("rel", tag="rel") or "rel"
            except Exception:
                anchor = "rel"

        tokens = ["why"]
        if anchor:
            tokens.append(anchor)
        tokens.extend([choice, P])

        if hasattr(self, "produce_utterance") and random.random() < 0.4:
            try:
                tokens.extend((self.produce_utterance() or "").split()[:1])
            except Exception:
                pass

        tokens = [t for t in tokens if isinstance(t, str)]
        proposal = " ".join(tokens)

        try:
            self.own_fitness += 0.12
        except Exception:
            pass

        if hasattr(self, "_observe_tokens"):
            try:
                self._observe_tokens(tokens, gain=0.07)
            except Exception:
                pass

        return {
            "agent_id": f"A{self.id}",
            "choice": choice,
            "utterance": proposal,
            "simA": simA,
            "simB": simB,
        }

    def _solve_describe_concept(self, task):
        """
        Task:
        task_type: "describe_concept"
        data: {"target_token": <tok>}

        Agent produces an utterance describing / circling the target token.
        Encourages noun-like anchors + property-ish neighbours.
        """
        data = task.get("data", {}) or {}
        target = data.get("target_token")

        sem = getattr(self, "semantic", {})
        vecs = sem.get("vecs", {})

        # Fallback if target missing
        if not target:
            if vecs:
                target = random.choice(list(vecs.keys()))
            else:
                target = "su"

        # Make sure we know about the token
        if hasattr(self, "_ensure_vec"):
            self._ensure_vec(target)

        # anchor concept token (ref/rel-like)
        anchor = None
        if hasattr(self, "_ensure_concept_token"):
            try:
                anchor = self._ensure_concept_token(str(target), tag="ref")
            except Exception:
                anchor = None

        # semantic neighbours of target
        neighbours = self.semantic_system._semantic_neighbors(target, k=3)

        # optional free tail
        tail = []
        if hasattr(self, "produce_utterance") and random.random() < 0.4:
            try:
                tail = (self.produce_utterance() or "").split()[:2]
            except Exception:
                tail = []

        toks = ["why"]
        if anchor:
            toks.append(anchor)
        toks.append(target)
        toks.extend(neighbours)
        toks.extend(tail)

        toks = [t for t in toks if isinstance(t, str) and t.strip()][:8]
        utter = " ".join(toks) if toks else target

        try:
            self.own_fitness += 0.10
        except Exception:
            pass

        if hasattr(self, "_observe_tokens"):
            try:
                self._observe_tokens(toks, gain=0.12)
            except Exception:
                pass

        return {
            "agent_id": f"A{self.id}",
            "mode": "describe",
            "target": target,
            "utterance": utter,
        }

    def _solve_action_reconstruction(self, task):
        """
        Task:
        task_type: "action_reconstruction"
        data: {"from": tokA, "to": tokB}

        Agent proposes an 'action' token that links A → B.
        Encourages verb-ish/process tokens and transform semantics.
        """
        data = task.get("data", {}) or {}
        a_tok = data.get("from", "su")
        b_tok = data.get("to", "tol")

        if hasattr(self, "_ensure_vec"):
            self._ensure_vec(a_tok)
            self._ensure_vec(b_tok)

        candidates = []

        # neighbours of A
        n_a = self.semantic_system._semantic_neighbors(a_tok, k=4)

        # neighbours of B
        n_b = self.semantic_system._semantic_neighbors(b_tok, k=4)

        # intersection first (tokens that “live” near both)
        inter = list(set(n_a) & set(n_b))
        if inter:
            candidates.extend(inter)
        else:
            candidates.extend(n_a[:2])
            candidates.extend(n_b[:2])

        # if nothing, invent a token via language organ
        act_tok = None
        for t in candidates:
            if isinstance(t, str) and t.strip():
                act_tok = t
                break

        if act_tok is None:
            if hasattr(self, "_invent_token"):
                try:
                    act_tok = self._invent_token(prefix="muk")
                except Exception:
                    act_tok = "muk"
            else:
                act_tok = "muk"

        # build utterance: 'why' + action + A + B + tiny tail
        tail = []
        if hasattr(self, "produce_utterance") and random.random() < 0.4:
            try:
                tail = (self.produce_utterance() or "").split()[:1]
            except Exception:
                tail = []

        toks = ["why", act_tok, a_tok, b_tok] + tail
        toks = [t for t in toks if isinstance(t, str) and t.strip()][:8]
        utter = " ".join(toks)

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
            "action_token": act_tok,
            "from": a_tok,
            "to": b_tok,
            "utterance": utter,
        }

    def _solve_similarity_debate(self, task):
        """
        Task:
        task_type: "similarity_debate"
        data: {"A": tok1, "B": tok2, "C": tok3}

        Agent picks which pair is closest in semantic space and
        produces an utterance referencing them.
        """
        data = task.get("data", {}) or {}
        A = data.get("A", "su")
        B = data.get("B", "tol")
        C = data.get("C", "muk")

        pairs = [(A, B), (A, C), (B, C)]
        best_pair = pairs[0]
        best_score = None

        if hasattr(self, "semantic_distance"):
            try:
                for x, y in pairs:
                    d = self.semantic_distance(x, y)
                    s = -d  # smaller distance = higher score
                    if best_score is None or s > best_score:
                        best_score = s
                        best_pair = (x, y)
            except Exception:
                pass

        x, y = best_pair

        anchor = None
        if hasattr(self, "_ensure_concept_token"):
            try:
                anchor = self._ensure_concept_token("similar", tag="rel")
            except Exception:
                anchor = None

        tail = []
        if hasattr(self, "produce_utterance") and random.random() < 0.4:
            try:
                tail = (self.produce_utterance() or "").split()[:2]
            except Exception:
                tail = []

        toks = ["why"]
        if anchor:
            toks.append(anchor)
        toks.extend([x, y])
        toks.extend(tail)
        toks = [t for t in toks if isinstance(t, str) and t.strip()][:8]
        utter = " ".join(toks)

        try:
            self.own_fitness += 0.10
        except Exception:
            pass

        if hasattr(self, "_observe_tokens"):
            try:
                self._observe_tokens(toks, gain=0.10)
            except Exception:
                pass

        return {
            "agent_id": f"A{self.id}",
            "chosen_pair": [x, y],
            "utterance": utter,
        }
