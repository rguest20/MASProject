"""
agents/cognition/task_system_v2.py
Task solving system (v2).
"""

import random
import re

from agents.agent_constants import REL_GT, REL_LT, REL_EQ


class TaskSystemV2:
    """
    Task system that:
      - keeps numeric help/teaching
      - uses emergent 'concept tokens' for explanations:
            why <rel-token> <free-language-tail>
      - no English vocabulary
      - concept tokens come from language_mixin_v2
      - supports cooperative numeric comparison tasks
    """

    def __init__(self, owner):
        object.__setattr__(self, "owner", owner)

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def __setattr__(self, name, value):
        if name == "owner":
            object.__setattr__(self, name, value)
            return
        setattr(self.owner, name, value)

    # ------------------------------------------------------
    # INITIALISATION
    # ------------------------------------------------------
    def _init_task_system(self):
        # base task state
        self.solved_tasks = {}
        self.failed_tasks = {}
        self.last_task_gen = -1
        self.tasks_attempted_this_gen = set()
        self.last_teach_seen = 0

        # optional logging hooks
        self.last_coop_log_gen = -1

        # dialogue bookkeeping
        self.last_dialogue_proposal = None

    def _task_fail(self, task_id):
        """Record a failed task attempt on the owning agent."""
        if task_id is not None:
            self.failed_tasks[task_id] = self.failed_tasks.get(task_id, 0) + 1
        if hasattr(self, "state"):
            self.state["frustration"] = min(
                1.0,
                self.state.get("frustration", 0.0) + 0.05,
            )
        return None

    # ------------------------------------------------------
    # ENTRY POINT
    # ------------------------------------------------------
    def try_solve_tasks(self, task_list, generation_index):
        # Always keep numeric help / teaching / family broadcast alive
        try:
            if hasattr(self, "maybe_answer_numeric_help"):
                self.maybe_answer_numeric_help()
        except Exception:
            pass

        try:
            if hasattr(self, "process_numeric_teaching"):
                self.process_numeric_teaching()
        except Exception:
            pass

        try:
            if hasattr(self, "maybe_broadcast_families"):
                self.maybe_broadcast_families()
        except Exception:
            pass

        if not task_list:
            return

        # generation reset
        if self.last_task_gen != generation_index:
            self.last_task_gen = generation_index
            self.tasks_attempted_this_gen = set()

        my_id = getattr(self, "id", None)

        def _is_task_available(t):
            tid = t.get("task_id")
            if tid in self.tasks_attempted_this_gen:
                return False

            assigned = t.get("assigned_agents", None)
            if assigned is not None and my_id is not None:
                # only explicitly assigned agents may attempt
                return my_id in assigned

            return True

        candidates = [t for t in task_list if _is_task_available(t)]
        if not candidates:
            return

        task = random.choice(candidates)
        tid = task.get("task_id")
        if tid is None:
            return
        self.tasks_attempted_this_gen.add(tid)

        # already solved previously? skip
        if tid in self.solved_tasks:
            return

        ttype = task.get("task_type")
        resp = None

        try:
            # Dispatch by task_type
            if ttype == "compare_numbers":
                resp = self._solve_compare_numbers(task)
            elif ttype == "cooperative_compare_numbers":
                resp = self._solve_cooperative_compare_numbers(task)
            elif ttype == "reconcile_counts":
                resp = self._solve_reconcile_counts(task)
            elif ttype == "translate_number":
                resp = self._solve_reconcile_counts(task)
            elif ttype == "referential_signal":
                resp = self._solve_referential_signal(task)
            elif ttype == "semantic_alignment":
                resp = self._solve_semantic_gap(task)
            elif ttype == "agreement_dialogue":
                resp = self._solve_agreement_dialogue(task)
            elif ttype == "explain_partner":
                resp = self._solve_explain_partner_answer(task)
            elif ttype == "token_compress":
                resp = self._solve_token_compression(task)
            elif ttype == "pref_align":
                resp = self._solve_preference_alignment_dialogue(task)
            elif ttype == "describe_concept":
                resp = self._solve_describe_concept(task)
            elif ttype == "action_reconstruction":
                resp = self._solve_action_reconstruction(task)
            elif ttype == "similarity_debate":
                resp = self._solve_similarity_debate(task)
            elif ttype == "narrative_chain":
                resp = self._solve_narrative_chain(task)
            elif ttype == "role_assignment":
                resp = self._solve_role_assignment(task)
            elif ttype == "misunderstanding_detection":
                resp = self._solve_misunderstanding_detection(task)
            elif ttype == "property_attribution":
                resp = self._solve_property_attribution(task)
            elif ttype == "verb_noun_compat":
                resp = self._solve_verb_noun_compat(task)
            elif ttype == "definition_swap":
                resp = self._solve_definition_swap(task)
            elif ttype == "prediction_task":
                resp = self._solve_prediction_task(task)
            else:
                return
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

    # ------------------------------------------------------
    # TASK TYPE: compare_numbers  (single-agent)
    # ------------------------------------------------------
    def _solve_compare_numbers(self, task):
        """
        Single-agent numeric comparison:
        - decode A and B with the reference numeric system if provided
        - otherwise fall back to this agent's own map
        - if either fails, request numeric help
        - otherwise choose the larger (or A if equal)
        - justification uses emergent rel* concept tokens
        """
        data = task.get("data", {}) or {}
        A = data.get("A", "") or ""
        B = data.get("B", "") or ""

        inst = task.get("instruction", {}) or {}
        override_map = inst.get("symbol_map", None)
        override_base = inst.get("base", None)

        # decode using override if present, else local
        valA = self._decode_number_phrase(A, override_map, override_base)
        valB = self._decode_number_phrase(B, override_map, override_base)

        # 1) request help if number not understood
        if valA is None or valB is None:
            ask = A if len(A.split()) >= len(B.split()) else B
            if ask and hasattr(self, "maybe_request_numeric_help"):
                try:
                    self.maybe_request_numeric_help(ask, None)
                except Exception:
                    pass
            return self._task_fail(task.get("task_id"))

        # 2) compare → relation
        if valA > valB:
            relation = REL_GT
            answer = A
        elif valB > valA:
            relation = REL_LT
            answer = B
        else:
            relation = REL_EQ
            answer = A

        just = self._build_relation_justification(relation)

        if hasattr(self, "_observe_tokens"):
            try:
                self._observe_tokens(just.split(), gain=0.15)
            except Exception:
                pass

        # 5) relation-truth teaching (optional, no vocab)
        self._maybe_emit_relation_teach(A, B, relation)

        # 6) confidence shaping
        if relation == REL_EQ:
            conf = 0.45 + 0.1 * random.random()
        else:
            conf = 0.55 + 0.2 * random.random()

        if hasattr(self, "adjust_trust"):
            try:
                self.adjust_trust(self.id, +0.015, channel=4)
            except Exception:
                pass

        return {
            "agent_id": f"A{self.id}",
            "answer": answer,
            "justification": just,
            "confidence": round(conf, 3),
        }

    # ------------------------------------------------------
    # TASK TYPE: cooperative_compare_numbers
    # ------------------------------------------------------
    def _solve_cooperative_compare_numbers(self, task):
        """
        Cooperative variant:
          - only agents in task["assigned_agents"] can see it
          - each agent decides whether to:
                * answer directly
                * request numeric help
          - both still generate justifications with a relation concept token
        """
        data = task.get("data", {})
        A = data.get("A", "")
        B = data.get("B", "")

        assigned = task.get("assigned_agents", [])
        my_id = getattr(self, "id", None)

        # identify partner (if any)
        partner_id = None
        if my_id is not None and len(assigned) >= 2:
            for aid in assigned:
                if aid != my_id:
                    partner_id = aid
                    break

        # Decode both sides (may fail)
        inst = task.get("instruction", {})
        override_map = inst.get("symbol_map")
        override_base = inst.get("base")

        valA = self._decode_number_phrase(A, override_map, override_base)
        valB = self._decode_number_phrase(B, override_map, override_base)

        # --------------------------------------------------
        # 1) If we can't decode, decide whether to ask partner
        # --------------------------------------------------
        if valA is None or valB is None:
            coop_w = getattr(self, "cooperation_weight", 0.3)

            # partner trust (channel 4 competence-ish), if available
            partner_trust = 0.0
            tc = getattr(self, "trust_channels", None)
            if partner_id is not None and isinstance(tc, dict):
                ch = tc.get(partner_id, {})
                if isinstance(ch, dict):
                    partner_trust = (
                        0.4 * ch.get("affinity", 0.0)
                        + 0.3 * ch.get("reliability", 0.0)
                        + 0.3 * ch.get("competence", 0.0)
                    )

            if (
                partner_id is not None
                and hasattr(self, "maybe_request_numeric_help")
                and random.random() < coop_w * (0.5 + 0.5 * partner_trust)
            ):
                ask = A if len(A.split()) >= len(B.split()) else B
                if ask:
                    try:
                        self.maybe_request_numeric_help(ask, None)
                    except Exception:
                        pass
                    self._log_coop_event(
                        task,
                        event="coop_request_help",
                        extra=f"partner=A{partner_id} phrase='{ask}'",
                    )
                return self._task_fail(task.get("task_id"))

            # otherwise fall back to normal failure + help request
            ask = A if len(A.split()) >= len(B.split()) else B
            if ask and hasattr(self, "maybe_request_numeric_help"):
                try:
                    self.maybe_request_numeric_help(ask, None)
                except Exception:
                    pass
            return self._task_fail(task.get("task_id"))

        # --------------------------------------------------
        # 2) Decode succeeded: compare as usual
        # --------------------------------------------------
        if valA > valB:
            relation = REL_GT
            answer = A
        elif valB > valA:
            relation = REL_LT
            answer = B
        else:
            relation = REL_EQ
            answer = A

        # relation justification with concept-token
        just = self._build_relation_justification(relation)

        if hasattr(self, "_observe_tokens"):
            try:
                self._observe_tokens(just.split(), gain=0.18)
            except Exception:
                pass

        # --------------------------------------------------
        # 3) Cooperative trust shaping
        # --------------------------------------------------
        if partner_id is not None and hasattr(self, "adjust_trust"):
            try:
                self.adjust_trust(partner_id, +0.03, channel=4)
                self.adjust_trust(partner_id, +0.02, channel=2)
            except Exception:
                pass

        # mutual self-trust bump
        if hasattr(self, "adjust_trust"):
            try:
                self.adjust_trust(self.id, +0.02, channel=4)
            except Exception:
                pass

        # occasionally log a co-op success event
        self._log_coop_event(
            task,
            event="coop_solved",
            extra=f"partner=A{partner_id}" if partner_id is not None else "partner=None",
        )

        # relation teaching as in single-agent mode
        self._maybe_emit_relation_teach(A, B, relation)

        # confidence slightly higher in co-op context
        if relation == REL_EQ:
            conf = 0.50 + 0.12 * random.random()
        else:
            conf = 0.60 + 0.20 * random.random()

        return {
            "agent_id": f"A{self.id}",
            "answer": answer,
            "justification": just,
            "confidence": round(conf, 3),
        }

    # ------------------------------------------------------
    # HELPER: build "why <rel-token> tail..." justification
    # ------------------------------------------------------
    def _build_relation_justification(self, relation):
        """
        Build an emergent justification utterance for a relation ("gt"/"lt"/"eq"):

          - anchor on a rel* concept token
          - expand through rel_family and semantic neighbors
          - optionally pull in numeric tokens from the last comparison
          - optionally add a short free utterance tail
        """
        sem = getattr(self, "semantic", None)
        concept_tok = None

        # 1) relation → rel-concept anchor
        if hasattr(self, "_ensure_concept_token"):
            try:
                concept_tok = self._ensure_concept_token(relation, tag="rel")
            except Exception:
                concept_tok = None

        rel_chain = []
        if concept_tok and isinstance(concept_tok, str):
            rel_chain.append(concept_tok)

        # 2) rel-family variants (existing meta-family)
        rel_family_tokens = []
        if sem is not None:
            rel_family_tokens = list(sem.get("rel_family", {}).keys())

        # allow rel-family to replace/augment the raw concept token
        if rel_family_tokens:
            if random.random() < 0.7:
                rel_chain = [random.choice(rel_family_tokens)]
            if random.random() < 0.4 and len(rel_family_tokens) > 1:
                candidate = random.choice(rel_family_tokens)
                if candidate not in rel_chain:
                    rel_chain.append(candidate)

        # 3) semantic neighbors of the reasoning chain
        neighbor_tokens = []
        for t in rel_chain:
            try:
                nbrs = self.semantic_system._semantic_neighbors(t, k=3)
            except Exception:
                nbrs = []
            if nbrs:
                neighbor_tokens.append(random.choice(nbrs))

        # 4) numeric evidence from last numeric phrase (if any)
        numeric_tokens = []
        if hasattr(self, "_last_tokens") and hasattr(self, "symbol_map"):
            try:
                symvals = set(self.symbol_map.values())
                for t in (self._last_tokens or []):
                    if t in symvals:
                        numeric_tokens.append(t)
                numeric_tokens = numeric_tokens[:2]
            except Exception:
                numeric_tokens = []

        # 5) tiny emergent tail from general utterance generator
        tail_tokens = []
        if hasattr(self, "produce_utterance") and random.random() < 0.5:
            try:
                tail = self.produce_utterance() or ""
                tail_tokens = tail.split()[:2]
            except Exception:
                tail_tokens = []

        # 6) Assemble
        just_tokens = ["why"]
        just_tokens.extend(rel_chain)
        just_tokens.extend(neighbor_tokens)
        just_tokens.extend(tail_tokens)

        # filter & cap
        just_tokens = [
            t for t in just_tokens
            if isinstance(t, str) and t.strip()
        ]
        if not just_tokens:
            if hasattr(self, "produce_utterance"):
                try:
                    return self.produce_utterance() or "why"
                except Exception:
                    return "why"
            return "why"

        just_tokens = just_tokens[:8]
        return " ".join(just_tokens)

    # ------------------------------------------------------
    # HELPER: emit relation teaching line
    # ------------------------------------------------------
    def _maybe_emit_relation_teach(self, A, B, relation):
        if not hasattr(self, "api") or self.api is None:
            return
        if relation not in (REL_GT, REL_LT, REL_EQ):
            return

        if random.random() >= 0.25:
            return

        try:
            line = f"A{self.id} teach_relation {A} || {B} rel={relation}"
            if hasattr(self.api, "append_help_response"):
                self.api.append_help_response(line)
            else:
                self.api.append_text("/help_responses.txt", line + "\n", scope="world")
        except Exception:
            pass

    # ------------------------------------------------------
    # HELPER: simple logging hook for cooperative events
    # ------------------------------------------------------
    def _log_coop_event(self, task, event, extra=""):
        if not hasattr(self, "api") or self.api is None:
            return
        try:
            tid = task.get("task_id", "unknown")
            ttype = task.get("task_type", "unknown")
            line = (
                f"A{self.id} coop_event task={tid} type={ttype} "
                f"event={event} {extra}"
            )
            if hasattr(self.api, "append_text"):
                self.api.append_text("/coop_events.txt", line + "\n", scope="world")
        except Exception:
            pass

    # ------------------------------------------------------
    # numeric decoding helpers
    # ------------------------------------------------------
    def _decode_number_phrase(self, phrase, override_map=None, override_base=None):
        if not phrase:
            return None

        sym_map = override_map if override_map else self.symbol_map
        b = override_base if override_base else self.counting.base

        rev = {v: k for k, v in sym_map.items()}

        digits = []
        for t in phrase.split():
            if t not in rev:
                return None
            d = rev[t]
            if not (0 <= d < b):
                return None
            digits.append(d)

        out = 0
        for d in digits:
            out = out * b + d

        if not isinstance(out, int):
            return None

        return out

    def _digits_for_tokens(self, tokens):
        if not hasattr(self, "symbol_map"):
            return None
        rev = {v: k for k, v in self.symbol_map.items()}
        try:
            return [rev[t] for t in tokens]
        except KeyError:
            return None

    # ------------------------------------------------------
    # TASK TYPE: reconcile_counts
    # ------------------------------------------------------
    def _solve_reconcile_counts(self, task):
        """Interpret another agent's numeral and report a normalised value."""
        views = task.get("views", {}) or {}
        phrase = views.get(self.id, views.get(str(self.id), ""))
        if not phrase:
            return self._task_fail(task.get("task_id"))

        value = self._decode_number_phrase(phrase)
        confidence = 0.85
        if value is None:
            confidence = 0.20
            value = 0
            for tok in phrase.split():
                digit = self.numeric_system.decode_token(tok)
                if digit is None:
                    return self._task_fail(task.get("task_id"))
                value = value * self.numeric_system.base + digit

        return {
            "agent_id": f"A{self.id}",
            "normalized": value,
            "utterance": phrase,
            "confidence": confidence,
        }

    def _solve_referential_signal(self, task):
        """Interpret a peer's invented signal using this agent's lexicon."""
        data = task.get("data", {}) or {}
        signal = data.get("signal", "")
        referent = None
        for known_referent, known_signal in self.referent_lexicon.items():
            if known_signal == signal:
                referent = known_referent
                break

        return {
            "agent_id": f"A{self.id}",
            "signal": signal,
            "referent": referent,
            "confidence": 0.85 if referent is not None else 0.10,
        }

    # ------------------------------------------------------
    # SEMANTIC ALIGNMENT TASK
    # ------------------------------------------------------
    def _solve_semantic_gap(self, task):
        """
        Agent responds to a semantic-gap task:
          - token: the word to inspect
          - community_vec: proposed centroid
          - community_confidence: 0..1
        """
        data = task.get("data", {})
        token = data.get("token")
        c_vec = data.get("community_vec")
        c_conf = float(data.get("community_confidence", 0.0))

        if not token or c_vec is None:
            return None

        sem = getattr(self, "semantic", None)
        if not sem:
            return None

        vecs = sem.get("vecs", {})
        my_vec = vecs.get(token)

        # If we don't know this token yet, lightly adopt community meaning
        if my_vec is None:
            try:
                my_vec = [float(x) for x in c_vec]
                vecs[token] = my_vec[:]
            except Exception:
                return None

        # distance estimate
        try:
            import math
            diff = [a - b for a, b in zip(my_vec, c_vec)]
            dist = math.sqrt(sum(d*d for d in diff))
        except Exception:
            dist = 0.0

        # simple heuristic: if we're not too far, we agree; else misaligned
        agree = dist < 2.0 or c_conf > 0.7

        # vector update
        if agree:
            eta_low = 0.03
            eta_high = 0.25
            eta = eta_low + (eta_high - eta_low) * (c_conf ** 2)
            eta = max(eta_low, min(eta_high, eta))
            new_vec = [
                (1.0 - eta) * a + eta * b
                for a, b in zip(my_vec, c_vec)
            ]
        else:
            eta = min(0.005, 0.02 * (1.0 - c_conf))
            new_vec = [
                a + eta * (a - b)
                for a, b in zip(my_vec, c_vec)
            ]

        vecs[token] = new_vec

        # reward shaping
        base_reward = 0.0
        if agree:
            base_reward = 0.5 * c_conf
        else:
            base_reward = 0.15 * (1.0 - c_conf)

        try:
            self.own_fitness += base_reward
        except Exception:
            pass

        if hasattr(self, "remember_interaction"):
            try:
                self.remember_interaction(
                    partner_id=-1,  # -1 = "community"
                    outcome=base_reward,
                    offspring_success=None,
                    gen_index=getattr(self, "current_generation", 0),
                )
            except Exception:
                pass

        return {
            "token": token,
            "agree": agree,
            "distance": dist,
            "confidence": c_conf,
            "reward": base_reward,
        }

    # ------------------------------------------------------
    # AGREEMENT DIALOGUE TASK
    # ------------------------------------------------------
    def _solve_agreement_dialogue(self, task):
        """
        Dialogue task:
            - Two agents negotiate over a topic (two emergent tokens).
            - Success comes from participation & token reuse pressure.
        """

        if not task:
            return None

        my_id = getattr(self, "id", None)
        assigned = task.get("assigned_agents", [])
        if my_id not in assigned:
            return None

        topic = task.get("topic", "")
        tid = task.get("task_id")

        # split topic
        try:
            a_tok, b_tok = [t.strip() for t in topic.split("||")]
        except Exception:
            a_tok, b_tok = "su", "tol"

        # 1) Build my proposal: small utterance linking them
        anchor = None
        if hasattr(self, "_ensure_concept_token"):
            anchors = ["agree", "align", "relate", "bind"]
            base_anchor = random.choice(anchors)
            try:
                anchor = self._ensure_concept_token(base_anchor, tag="rel")
            except Exception:
                anchor = base_anchor

        # semantic neighbors add flavour
        nbrs = []
        try:
            nb = self.semantic_system._semantic_neighbors(anchor, k=2)
            if nb:
                nbrs.append(random.choice(nb))
        except Exception:
            pass

        # free-language tail
        tail = []
        if hasattr(self, "produce_utterance") and random.random() < 0.6:
            try:
                tail = (self.produce_utterance() or "").split()[:2]
            except Exception:
                tail = []

        # choose which topic tokens to emphasise
        candidates = [a_tok, b_tok]
        chosen = []
        sem = getattr(self, "semantic", {})
        vecs = sem.get("vecs", {})

        for tok in candidates:
            if tok in vecs:
                chosen.append(tok)

        if not chosen:
            chosen = [random.choice(candidates)]

        if len(chosen) == 2 and random.random() < 0.5:
            chosen = [random.choice(chosen)]

        my_tokens = ["why"]
        if anchor:
            my_tokens.append(anchor)
        my_tokens.extend(chosen)
        my_tokens.extend(nbrs)
        my_tokens.extend(tail)
        my_tokens = [t for t in my_tokens if isinstance(t, str) and t.strip()]
        my_tokens = my_tokens[:8]
        proposal = " ".join(my_tokens) if my_tokens else "su tol"

        self.last_dialogue_proposal = my_tokens

        # participation reward
        try:
            self.own_fitness += 0.15
        except Exception:
            pass

        # semantic learning
        if hasattr(self, "_observe_tokens"):
            try:
                self._observe_tokens(my_tokens, gain=0.1)
            except Exception:
                pass

        return {
            "agent_id": f"A{self.id}",
            "proposal": proposal,
            "reused_partner_tokens": False,  # left for future use
        }

    # ------------------------------------------------------
    # EXPLAIN-PARTNER TASK
    # ------------------------------------------------------
    def _solve_explain_partner_answer(self, task):
        """
        Explain-Partner task.
        Agent sees partner's answer (raw tokens) and must produce
        a paraphrase or interpretation using its own vocabulary.
        """

        if not task:
            return None

        my_id = getattr(self, "id", None)
        assigned = task.get("assigned_agents", [])
        if my_id not in assigned or len(assigned) != 2:
            return None

        partner_id = assigned[0] if assigned[1] == my_id else assigned[1]
        partner_answer = task.get("partner_answer", "")
        partner_tokens = partner_answer.split() if partner_answer else []

        # 1) Anchor: relation token
        anchor = None
        if hasattr(self, "_ensure_concept_token"):
            try:
                anchor = self._ensure_concept_token("rel", tag="rel") or "rel"
            except Exception:
                anchor = "rel"

        # 2) Try to reuse some partner tokens
        reused = []
        sem = getattr(self, "semantic", {})
        vecs = sem.get("vecs", {})

        for t in partner_tokens:
            if t in vecs and random.random() < 0.6:
                reused.append(t)

        # 3) Add my own flavour / structure
        structured = []
        if random.random() < 0.4 and anchor:
            structured.append(anchor)
        if hasattr(self, "produce_utterance") and random.random() < 0.3:
            try:
                structured.append((self.produce_utterance() or "").split()[0])
            except Exception:
                pass

        tokens = ["why"] + reused[:3] + structured[:3]
        tokens = [t for t in tokens if isinstance(t, str) and t.strip()]
        explanation = " ".join(tokens) if tokens else "why " + (reused[0] if reused else "su")

        # Fitness
        try:
            self.own_fitness += 0.1  # participation
            if reused:
                self.own_fitness += 0.1
        except Exception:
            pass

        if reused and hasattr(self, "adjust_trust"):
            try:
                self.adjust_trust(partner_id, +0.02, channel=2)
            except Exception:
                pass

        # Learning
        if hasattr(self, "_observe_tokens"):
            try:
                self._observe_tokens(tokens, gain=0.1)
            except Exception:
                pass

        return {
            "agent_id": f"A{self.id}",
            "explanation": explanation,
            "tokens_reused": reused,
        }

    # ------------------------------------------------------
    # TOKEN COMPRESSION / EXPANSION TASK
    # ------------------------------------------------------
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

    # ------------------------------------------------------
    # PREFERENCE ALIGNMENT DIALOGUE
    # ------------------------------------------------------
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

    # ======================================================
    # LANGUAGE EMERGENCE TASKS
    # ======================================================

    # ------------------------------
    # 1. DESCRIBE–CONCEPT TASK
    # ------------------------------
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

    # ------------------------------
    # 2. ACTION RECONSTRUCTION TASK
    # ------------------------------
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

    # ------------------------------
    # 3. SIMILARITY DEBATE TASK
    # ------------------------------
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

    # ------------------------------
    # 4. NARRATIVE CHAIN TASK
    # ------------------------------
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

    # ------------------------------
    # 5. ROLE ASSIGNMENT TASK
    # ------------------------------
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

    # ------------------------------
    # 6. MISUNDERSTANDING DETECTION TASK
    # ------------------------------
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

    # ------------------------------
    # 7. PROPERTY ATTRIBUTION TASK
    # ------------------------------
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

    # ------------------------------
    # 8. VERB–NOUN COMPATIBILITY TASK
    # ------------------------------
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

    # ------------------------------
    # 9. DEFINITION SWAP TASK
    # ------------------------------
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

    # ------------------------------
    # 10. PREDICTION TASK
    # ------------------------------
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
