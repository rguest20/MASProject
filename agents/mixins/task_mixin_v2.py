# agents/mixins/task_mixin_v2.py
import random
import re

from agents.agent_constants import REL_GT, REL_LT, REL_EQ
from agents.mixins.task_mixin import TaskMixin


class TaskMixinV2(TaskMixin):
    """
    Task system that:
      - keeps numeric help/teaching
      - uses emergent 'concept tokens' for explanations:
            why <concept-token> <free-language-tail>
      - no English vocabulary
      - concept tokens come from language_mixin_v2
      - supports cooperative numeric comparison tasks
    """

    # ------------------------------------------------------
    # INITIALISATION
    # ------------------------------------------------------
    def _init_task_system(self):
        # base TaskMixin init (solved/failed/etc.)
        self.solved_tasks = {}
        self.failed_tasks = {}
        self.last_task_gen = -1
        self.tasks_attempted_this_gen = set()
        self.last_teach_seen = 0

        # for logging co-op choices if desired
        self.last_coop_log_gen = -1

        self.last_dialogue_proposal = None

    # ------------------------------------------------------
    # ENTRY POINT
    # ------------------------------------------------------
    def try_solve_tasks(self, task_list, generation_index):

        # always process numeric help + teaching + family broadcast
        if hasattr(self, "maybe_answer_numeric_help"):
            self.maybe_answer_numeric_help()
        if hasattr(self, "process_numeric_teaching"):
            self.process_numeric_teaching()
        if hasattr(self, "maybe_broadcast_families"):
            self.maybe_broadcast_families()

        if not task_list:
            return

        # generation reset
        if self.last_task_gen != generation_index:
            self.last_task_gen = generation_index
            self.tasks_attempted_this_gen = set()

        # --------------------------------------------------
        # Filter tasks by:
        #   - not attempted this gen
        #   - (optional) assigned_agents, if present
        # --------------------------------------------------
        my_id = getattr(self, "id", None)

        def _is_task_available(t):
            tid = t.get("task_id")
            if tid in self.tasks_attempted_this_gen:
                return False

            assigned = t.get("assigned_agents", None)
            if assigned is not None and my_id is not None:
                # only agents explicitly assigned may attempt
                return my_id in assigned

            return True

        candidates = [t for t in task_list if _is_task_available(t)]
        if not candidates:
            return

        task = random.choice(candidates)
        tid = task["task_id"]
        self.tasks_attempted_this_gen.add(tid)

        # Already solved previously?
        if tid in self.solved_tasks:
            return

        ttype = task.get("task_type")

        # --------------------------------------------------
        # Dispatch by task_type
        # --------------------------------------------------
        if ttype == "compare_numbers":
            resp = self._solve_compare_numbers(task)
        elif ttype == "cooperative_compare_numbers":
            resp = self._solve_cooperative_compare_numbers(task)
        elif ttype == "semantic_alignment":
            resp = self._solve_semantic_gap(task)
        elif ttype == "agreement_dialogue":
            resp = self._solve_agreement_dialogue(task)
        elif ttype == "explain_partner":
            return self._solve_explain_partner_answer(task)
        elif ttype == "token_compress":
            return self._solve_token_compression(task)
        elif ttype == "pref_align":
            return self._solve_preference_alignment_dialogue(task)
        else:
            return

        # record and semantically reinforce justification tokens
        if resp:
            self.solved_tasks[tid] = resp
            if hasattr(self, "_observe_tokens"):
                toks = resp.get("justification", "").split()
                if toks:
                    self._observe_tokens(toks, gain=0.2)

    # ------------------------------------------------------
    # TASK TYPE: compare_numbers  (single-agent)
    # ------------------------------------------------------
    def _solve_compare_numbers(self, task):
        data = task["data"]
        A = data["A"]
        B = data["B"]

        valA = self._decode_number_phrase(A)
        valB = self._decode_number_phrase(B)

        # 1) request help if number not understood
        if valA is None or valB is None:
            ask = A if len(A.split()) >= len(B.split()) else B
            if ask and hasattr(self, "maybe_request_numeric_help"):
                self.maybe_request_numeric_help(ask, None)
            return self._task_fail(task["task_id"])

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
            self._observe_tokens(just.split(), gain=0.15)

        # 5) relation-truth teaching (optional, no vocab)
        self._maybe_emit_relation_teach(A, B, relation)

        # 6) confidence shaping
        if relation == REL_EQ:
            conf = 0.45 + 0.1 * random.random()
        else:
            conf = 0.55 + 0.2 * random.random()

        if hasattr(self, "adjust_trust"):
            self.adjust_trust(self.id, +0.015, channel=4)

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
                * request numeric help (which uses trust + cooperation_weight)
          - both still generate justifications with a relation concept token
        """
        data = task["data"]
        A = data["A"]
        B = data["B"]

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
        valA = self._decode_number_phrase(A)
        valB = self._decode_number_phrase(B)

        # --------------------------------------------------
        # 1) If we can't decode, decide whether to ask partner
        # --------------------------------------------------
        if valA is None or valB is None:
            # cooperation_weight biases asking for help instead of
            # silently failing
            coop_w = getattr(self, "cooperation_weight", 0.3)

            # partner trust (channel 4 competence-ish)
            partner_trust = 0.0
            if partner_id is not None:
                ch = getattr(self, "trust_channels", {}).get(partner_id, {})
                partner_trust = (
                    0.4 * ch.get("affinity", 0.0)
                    + 0.3 * ch.get("reliability", 0.0)
                    + 0.3 * ch.get("competence", 0.0)
                )

            # simple decision: if we have a partner and cooperation/ trust
            # is high enough, ask them rather than guess
            if (
                partner_id is not None
                and hasattr(self, "maybe_request_numeric_help")
                and random.random() < coop_w * (0.5 + 0.5 * partner_trust)
            ):
                ask = A if len(A.split()) >= len(B.split()) else B
                if ask:
                    self.maybe_request_numeric_help(ask, None)
                    self._log_coop_event(
                        task,
                        event="coop_request_help",
                        extra=f"partner=A{partner_id} phrase='{ask}'",
                    )
                return self._task_fail(task["task_id"])

            # otherwise fall back to normal failure
            ask = A if len(A.split()) >= len(B.split()) else B
            if ask and hasattr(self, "maybe_request_numeric_help"):
                self.maybe_request_numeric_help(ask, None)
            return self._task_fail(task["task_id"])

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
            self._observe_tokens(just.split(), gain=0.18)

        # --------------------------------------------------
        # 3) Cooperative trust shaping
        # --------------------------------------------------
        if partner_id is not None and hasattr(self, "adjust_trust"):
            # reward partner with a small competence/affinity bump
            self.adjust_trust(partner_id, +0.03, channel=4)
            self.adjust_trust(partner_id, +0.02, channel=2)

        # mutual self-trust bump
        if hasattr(self, "adjust_trust"):
            self.adjust_trust(self.id, +0.02, channel=4)

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
    # HELPER: build "why <concept-token> tail..." justification
    # ------------------------------------------------------
    def _build_relation_justification(self, relation):
        """
        Build an emergent justification utterance for a relation ("gt"/"lt"/"eq"):

          - anchor on a rel* concept token
          - expand through rel_family and semantic neighbors
          - optionally pull in numeric tokens from the last comparison
          - optionally add a short free utterance tail

        Output is a short sequence like:
            "why relvakrinrin belmukzev zevtoltol ..."

        All pieces are drawn from the agent's own lexicon + semantics.
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
            # often start from a "real" rel-family item
            if random.random() < 0.7:
                rel_chain = [random.choice(rel_family_tokens)]
            # sometimes add a second rel-token for richness
            if random.random() < 0.4 and len(rel_family_tokens) > 1:
                candidate = random.choice(rel_family_tokens)
                if candidate not in rel_chain:
                    rel_chain.append(candidate)

        # 3) semantic neighbors of the reasoning chain
        neighbor_tokens = []
        if hasattr(self, "_semantic_neighbors"):
            for t in rel_chain:
                nbrs = self._semantic_neighbors(t, k=3)
                if nbrs:
                    neighbor_tokens.append(random.choice(nbrs))

        # 4) numeric evidence from last numeric phrase (if any)
        numeric_tokens = []
        if hasattr(self, "_last_tokens") and hasattr(self, "symbol_map"):
            symvals = set(self.symbol_map.values())
            for t in (self._last_tokens or []):
                if t in symvals:
                    numeric_tokens.append(t)
            numeric_tokens = numeric_tokens[:2]

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
        just_tokens.extend(numeric_tokens)
        just_tokens.extend(tail_tokens)

        # filter & cap
        just_tokens = [
            t for t in just_tokens
            if isinstance(t, str) and t.strip()
        ]
        if not just_tokens:
            # degenerate case → fully emergent utterance
            if hasattr(self, "produce_utterance"):
                return self.produce_utterance() or "why"
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
        """
        Writes a one-line log for cooperative tasks so you can
        verify they’re being used as intended.
        """
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
    # numeric decoding helpers (unchanged)
    # ------------------------------------------------------
    def _decode_number_phrase(self, phrase):
        if not phrase:
            return None
        if not hasattr(self, "symbol_map") or not hasattr(self, "counting"):
            return None

        rev = {v: k for k, v in self.symbol_map.items()}
        base = self.counting.base

        digits = []
        for t in phrase.split():
            if t not in rev:
                return None
            d = rev[t]
            if d < 0 or d >= base:
                return None
            digits.append(d)

        out = 0
        for d in digits:
            out = out * base + d
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
    # SEMANTIC ALIGNMENT TASK
    # ------------------------------------------------------
    def _solve_semantic_gap(self, task):
        """
        Agent responds to a semantic-gap task:
          - token: the word to inspect
          - community_vec: proposed centroid
          - community_confidence: 0..1
        Mixed-mode behaviour:
          - if community confidence is high, aligning pulls our vector strongly
          - if low, aligning is weaker (we keep exploring)
        """
        data = task.get("data", {})
        token = data.get("token")
        c_vec = data.get("community_vec")
        c_conf = float(data.get("community_confidence", 0.0))

        if not token or c_vec is None:
            return None  # cannot answer

        sem = getattr(self, "semantic", None)
        if not sem:
            return None

        vecs = sem.get("vecs", {})
        my_vec = vecs.get(token)
        if my_vec is None:
            # If we don't know this token yet, lightly adopt community meaning
            try:
                my_vec = [float(x) for x in c_vec]
                vecs[token] = my_vec[:]
            except Exception:
                return None

        # --- Decide whether we "agree" or "disagree" ---
        # simple heuristic: if we're not too far, we agree; else we mark it as misaligned
        try:
            import math
            diff = [a - b for a, b in zip(my_vec, c_vec)]
            dist = math.sqrt(sum(d*d for d in diff))
        except Exception:
            dist = 0.0

        agree = dist < 2.0 or c_conf > 0.7

        # --- Vector update (Mode C) ---
        # base learning rate is small; boosted for high-confidence tokens
        if agree:
            # move toward community
            eta_low = 0.03
            eta_high = 0.25
            eta = eta_low + (eta_high - eta_low) * (c_conf ** 2)
            eta = max(eta_low, min(eta_high, eta))
            new_vec = [
                (1.0 - eta) * a + eta * b
                for a, b in zip(my_vec, c_vec)
            ]
        else:
            # if we explicitly disagree, we only make a tiny self-consistency tweak
            # (keeps diversity alive while not exploding)
            eta = 0.02 * (1.0 - c_conf)
            new_vec = [
                a + eta * (a - b)
                for a, b in zip(my_vec, c_vec)
            ]

        sem["vecs"][token] = new_vec

        # --- Reward shaping back to fitness / social memory ---
        # reward more when aligning with high-confidence community tokens
        base_reward = 0.0
        if agree:
            base_reward = 0.5 * c_conf
        else:
            # a little reward for resisting low-confidence majority
            base_reward = 0.15 * (1.0 - c_conf)

        # fold into own_fitness and trust
        try:
            self.own_fitness += base_reward
        except Exception:
            pass

        # record that we engaged with a community task
        if hasattr(self, "remember_interaction"):
            self.remember_interaction(
                partner_id=-1,  # -1 = "community"
                outcome=base_reward,
                offspring_success=None,
                gen_index=getattr(self, "current_generation", 0),
            )

        # Minimal textual "explanation" if you log responses
        return {
            "token": token,
            "agree": agree,
            "distance": dist,
            "confidence": c_conf,
            "reward": base_reward,
        }

    def _solve_agreement_dialogue(self, task):
        """
        Dialogue task:
            - Two agents negotiate over a topic (two emergent tokens).
            - Success comes from: participation, reuse of partner tokens,
            and converging toward an agreed relation/concept.
            - There is *no correct answer*; it's a pressure toward shared structure.
        """

        if not task:
            return None

        topic = task.get("topic", "")
        assigned = task.get("assigned_agents", [])
        tid = task.get("task_id")
        my_id = getattr(self, "id", None)

        if my_id not in assigned:
            return None

        # identify partner
        partner = None
        if len(assigned) == 2:
            partner = assigned[0] if assigned[1] == my_id else assigned[1]

        # split topic
        try:
            a_tok, b_tok = [t.strip() for t in topic.split("||")]
        except:
            a_tok, b_tok = "su", "tol"

        # ---------------------------------------------------
        # 1) Produce my proposal: a small utterance linking them
        # ---------------------------------------------------
        # Try to build a relation-like chain using existing machinery
        anchor = None
        if hasattr(self, "_ensure_concept_token"):
            # choose a *specific* conceptual anchor (agree / contrast / relate / bind)
            anchors = ["agree", "align", "relate", "bind"]
            base_anchor = random.choice(anchors)

            try:
                anchor = self._ensure_concept_token(base_anchor, tag="rel")
            except Exception:
                anchor = base_anchor

        # semantic neighbors add flavour
        nbrs = []
        if hasattr(self, "_semantic_neighbors") and anchor:
            try:
                nb = self._semantic_neighbors(anchor, k=2)
                if nb:
                    nbrs.append(random.choice(nb))
            except Exception:
                pass

        # free-language tail for creativity
        tail = ""
        if hasattr(self, "produce_utterance") and random.random() < 0.6:
            tail = (self.produce_utterance() or "").split()[:2]
        else:
            tail = []

        # assemble proposal
        # choose which token(s) from the topic to emphasise
        candidates = [a_tok, b_tok]
        chosen = []

        # bias toward tokens they have seen before or have vectors for
        sem = getattr(self, "semantic", {})
        vecs = sem.get("vecs", {})

        for tok in candidates:
            if tok in vecs:
                chosen.append(tok)

        # ensure at least one choice even if unknown
        if not chosen:
            chosen = [random.choice(candidates)]

        # sometimes emphasise only one
        if len(chosen) == 2 and random.random() < 0.5:
            chosen = [random.choice(chosen)]

        my_tokens = ["why", anchor] + chosen
        my_tokens.extend(nbrs)
        my_tokens.extend(tail)
        my_tokens = [t for t in my_tokens if isinstance(t, str) and t.strip()]
        my_tokens = my_tokens[:8]
        proposal = " ".join(my_tokens) if my_tokens else "su tol"

        self.last_dialogue_proposal = my_tokens

        # ---------------------------------------------------
        # 2) Reward participation immediately
        # ---------------------------------------------------
        try:
            self.own_fitness += 0.15   # mild reward for engaging
        except Exception:
            pass

        # ---------------------------------------------------
        # 3) Try to detect partner reuse (agreement pressure)
        # ---------------------------------------------------
        reused = False
        if partner is not None:
            p = next((ag for ag in self.population if ag.id == partner), None)
            if p and hasattr(p, "last_dialogue_proposal") and p.last_dialogue_proposal:
                for tok in p.last_dialogue_proposal:
                    if tok in chosen:    # only compare chosen tokens
                        reused = True
                        break

        # ---------------------------------------------------
        # 4) Update semantic vectors lightly toward used tokens
        # ---------------------------------------------------
        if hasattr(self, "_observe_tokens"):
            self._observe_tokens(my_tokens, gain=0.1)

        # 4b) Light semantic convergence
        # Move chosen tokens' vectors slightly toward each other
        if sem and "vecs" in sem and len(chosen) >= 1:
            vecs = sem["vecs"]
            base = vecs.get(chosen[0])
            if base:
                for tok in chosen[1:]:
                    if tok in vecs:
                        v = vecs[tok]
                        vecs[tok] = [
                            v[i] + 0.05 * (base[i] - v[i])
                            for i in range(len(v))
                        ]

        # ---------------------------------------------------
        # 5) Produce output (mostly for logs)
        # ---------------------------------------------------
        return {
            "agent_id": f"A{self.id}",
            "proposal": proposal,
            "reused_partner_tokens": reused,
        }

    def _solve_explain_partner_answer(self, task):
        """
        Explain-Partner task.
        Agent sees partner's answer (raw tokens) and must produce
        a paraphrase or interpretation using its own vocabulary.

        Fitness reward comes from:
        - Participation
        - Reusing tokens from partner (alignment)
        - Adding structured relations (for explorers)
        """

        if not task:
            return None

        my_id = getattr(self, "id", None)
        assigned = task.get("assigned_agents", [])
        if my_id not in assigned:
            return None

        partner_id = assigned[0] if assigned[1] == my_id else assigned[1]
        partner_answer = task.get("partner_answer", "")
        partner_tokens = partner_answer.split() if partner_answer else []

        # -----------------------------
        # Build my explanation
        # -----------------------------
        # 1) Anchor: relation token
        anchor = None
        if hasattr(self, "_ensure_concept_token"):
            anchor = self._ensure_concept_token("rel", tag="rel") or "rel"

        # 2) Try to reuse some partner tokens
        reused = []
        for t in partner_tokens:
            if t in self.semantic.get("vecs", {}) and random.random() < 0.6:
                reused.append(t)

        # 3) Add my own flavour / structure
        structured = []
        if random.random() < 0.4:
            structured.append(anchor or "rel")
        if random.random() < 0.3:
            structured.append(self.produce_utterance().split()[0])

        tokens = ["why"] + reused[:3] + structured[:3]
        tokens = [t for t in tokens if isinstance(t, str) and t.strip()]
        explanation = " ".join(tokens) if tokens else "why " + (reused[0] if reused else "su")

        # -----------------------------
        # Fitness
        # -----------------------------
        self.own_fitness += 0.1  # participation

        if reused:
            self.own_fitness += 0.1
            if hasattr(self, "adjust_trust"):
                self.adjust_trust(partner_id, +0.02, channel=2)

        # -----------------------------
        # Learning
        # -----------------------------
        if hasattr(self, "_observe_tokens"):
            self._observe_tokens(tokens, gain=0.1)

        return {
            "agent_id": f"A{self.id}",
            "explanation": explanation,
            "tokens_reused": reused,
        }

    def _solve_token_compression(self, task):
        """
        Token-Compression / Expansion task.
        If the input utterance is long: compress it.
        If short: expand it.

        This encourages:
        - H-types to summarise
        - E-types to elaborate
        """

        if not task:
            return None

        my_id = getattr(self, "id", None)
        assigned = task.get("assigned_agents", [])
        if my_id not in assigned:
            return None

        utt = task.get("utterance", "")
        toks = utt.split()

        # -----------------------------
        # Determine mode
        # -----------------------------
        if len(toks) > 6:
            mode = "compress"
        else:
            mode = "expand"

        # -----------------------------
        # Compression: pick salient roots
        # -----------------------------
        if mode == "compress":
            roots = []
            for t in toks:
                if t in self.semantic.get("vecs", {}) and random.random() < 0.25:
                    roots.append(t)
            if not roots:
                roots = toks[:2]

            # produce a 2–4 token summary
            summary = " ".join(roots[:4])

            self.own_fitness += 0.15  # rewarded for condensation
            if hasattr(self, "_observe_tokens"):
                self._observe_tokens(roots, gain=0.05)

            return {
                "agent_id": f"A{self.id}",
                "mode": "compress",
                "result": summary,
            }

        # -----------------------------
        # Expansion: add relations + neighbours
        # -----------------------------
        else:
            anchor = None
            if hasattr(self, "_ensure_concept_token"):
                anchor = self._ensure_concept_token("rel", tag="rel") or "rel"

            expanded = ["why", anchor] if anchor else ["why"]

            # pull semantic neighbours for elaboration
            if hasattr(self, "_semantic_neighbors"):
                for t in toks:
                    try:
                        nbs = self._semantic_neighbors(t, k=1)
                        if nbs:
                            expanded.append(nbs[0])
                    except:
                        pass

            # free-language creativity tail
            if hasattr(self, "produce_utterance") and random.random() < 0.7:
                expanded.extend((self.produce_utterance() or "").split()[:2])

            # trim to sane size
            expanded = [x for x in expanded if isinstance(x, str)][:8]
            sentence = " ".join(expanded)

            self.own_fitness += 0.15  # reward elaboration
            if hasattr(self, "_observe_tokens"):
                self._observe_tokens(expanded, gain=0.05)

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
        the *agreement* matters more than correctness.

        Pressure:
        - semantic similarity
        - partner reuse
        - relation anchoring
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

        # compute similarity scores if available
        simA = 0.0
        simB = 0.0
        if hasattr(self, "_semantic_similarity"):
            try:
                simA = self._semantic_similarity(A, P)
                simB = self._semantic_similarity(B, P)
            except:
                pass

        choice = A if simA >= simB else B

        # produce a reasoned utterance
        anchor = None
        if hasattr(self, "_ensure_concept_token"):
            anchor = self._ensure_concept_token("rel", tag="rel") or "rel"

        tokens = ["why", anchor, choice, P]
        if hasattr(self, "produce_utterance") and random.random() < 0.4:
            tokens.extend((self.produce_utterance() or "").split()[:1])

        tokens = [t for t in tokens if isinstance(t, str)]
        proposal = " ".join(tokens)

        # reward participation
        self.own_fitness += 0.12

        # alignment: if partner chose same, coordinator will reinforce trust
        if hasattr(self, "_observe_tokens"):
            self._observe_tokens(tokens, gain=0.07)

        return {
            "agent_id": f"A{self.id}",
            "choice": choice,
            "utterance": proposal,
            "simA": simA,
            "simB": simB,
        }