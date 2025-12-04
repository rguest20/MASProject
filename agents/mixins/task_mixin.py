# agents/mixins/task_mixin.py
import random
import re


class TaskMixin:

    # ------------------------------------------------------
    # INITIALISATION
    # ------------------------------------------------------
    def _init_task_system(self):
        self.solved_tasks = {}           # task_id → response object
        self.failed_tasks = {}           # task_id → failure count
        self.last_task_gen = -1
        self.tasks_attempted_this_gen = set()

        # how many lines of /help_responses.txt we’ve processed
        self.last_teach_seen = 0

        # for operator-fitness reward
        self.last_relation_event = None

    # ------------------------------------------------------
    # ENTRY POINT: called by Coordinator each generation
    # ------------------------------------------------------
    def try_solve_tasks(self, task_list, generation_index):
        """
        Agents solve AT MOST one task per generation,
        and never solve the same task twice.
        """

        # Always process incoming help + teaching each gen
        if hasattr(self, "maybe_answer_numeric_help"):
            self.maybe_answer_numeric_help()
        if hasattr(self, "process_numeric_teaching"):
            self.process_numeric_teaching()

        if not task_list:
            return

        # generation reset
        if self.last_task_gen != generation_index:
            self.last_task_gen = generation_index
            self.tasks_attempted_this_gen = set()

        # pick unseen tasks for this generation
        candidates = [
            t for t in task_list
            if t["task_id"] not in self.tasks_attempted_this_gen
        ]
        if not candidates:
            return

        task = random.choice(candidates)
        tid = task["task_id"]
        self.tasks_attempted_this_gen.add(tid)

        # Already solved previously?
        if tid in self.solved_tasks:
            return

        # solve
        if task["task_type"] == "compare_numbers":
            resp = self._solve_compare_numbers(task)
        else:
            return

        if resp:
            self.solved_tasks[tid] = resp

            # ✅ Semantic reinforcement from justification tokens
            if hasattr(self, "_observe_tokens"):
                just = resp.get("justification", "") or ""
                toks = just.replace("_", " ").split()
                if toks:
                    self._observe_tokens(toks, gain=0.2)

    # ------------------------------------------------------
    # TASK TYPE: number comparison
    # ------------------------------------------------------
    def _solve_compare_numbers(self, task):
        tdata = task["data"]
        A = tdata["A"]
        B = tdata["B"]

        valA = self._decode_number_phrase(A)
        valB = self._decode_number_phrase(B)

        # request help if decode fails
        if valA is None or valB is None:
            ask = A if len(A.split()) >= len(B.split()) else B
            if ask:
                self.maybe_request_numeric_help(ask, None)
            return self._task_fail(task["task_id"])

        # ---------------------------------------------------
        # Compare & orient: LEFT is the chosen (larger/equal)
        # ---------------------------------------------------
        if valA > valB:
            answer = A
            relation = "gt"
            left_tokens  = A.split()
            right_tokens = B.split()
        elif valB > valA:
            answer = B
            relation = "lt"
            left_tokens  = B.split()
            right_tokens = A.split()
        else:
            answer = A
            relation = "eq"
            left_tokens  = A.split()
            right_tokens = B.split()

        # ===================================================
        #   EMERGENT JUSTIFICATION (NO HUMAN VOCABULARY)
        # ===================================================
        #
        # Agents generate their own "reason token sequence".
        # This is evolution’s playground: they invent tokens,
        # grammar, and structure over generations.
        #
        # ---------------------------------------------------

        # The agent’s language engine – emergent symbols only
        if hasattr(self, "produce_utterance"):
            # A single utterance per justification (minimal pressure)
            just = self.produce_utterance()
        else:
            just = ""

        # If empty / failed, fall back to random tokens
        if not just:
            toks = []
            for _ in range(random.randint(1, 3)):
                toks.append(self.produce_utterance())
            just = " ".join(toks)

        # Feed emergent tokens into semantic system
        if hasattr(self, "_observe_tokens"):
            self._observe_tokens(just.split(), gain=0.15)

        # ===================================================
        #   RELATION TEACHING  (truth signal, not vocabulary)
        # ===================================================
        #
        # We still allow agents to emit:
        #    "teach_relation A || B rel=gt"
        #
        # to give the *correct answer*, but NOT the reasoning.
        # This encourages learning, not imitation.
        #
        # Later we can remove this too.
        # ---------------------------------------------------

        if hasattr(self, "api") and self.api is not None:
            if random.random() < 0.25:  # reduced frequency to avoid dominance
                try:
                    teach_line = f"A{self.id} teach_relation {A} || {B} rel={relation}"
                    if hasattr(self.api, "append_help_response"):
                        self.api.append_help_response(teach_line)
                    else:
                        self.api.append_text("/help_responses.txt", teach_line + "\n", scope="world")
                except Exception:
                    pass

        # --------------------------------
        # Confidence shaping
        # --------------------------------
        if relation == "eq":
            conf = 0.45 + 0.1 * random.random()
        else:
            conf = 0.55 + 0.2 * random.random()

        # Self-trust reward
        if hasattr(self, "adjust_trust"):
            self.adjust_trust(self.id, +0.015, channel=4)

        return {
            "agent_id": f"A{self.id}",
            "answer": answer,
            "justification": just,
            "confidence": round(conf, 3),
        }

    # ------------------------------------------------------
    def _task_fail(self, tid):
        self.failed_tasks[tid] = self.failed_tasks.get(tid, 0) + 1

        if hasattr(self, "state"):
            S = self.state
            S["frustration"] = min(1.0, S["frustration"] + 0.05)

        return None

    # ------------------------------------------------------
    # NUMBER DECODING
    # ------------------------------------------------------
    def _decode_number_phrase(self, phrase):
        if not phrase:
            return None

        if not hasattr(self, "symbol_map") or not hasattr(self, "counting"):
            return None

        rev = {v: k for k, v in self.symbol_map.items()}
        base = self.counting.base

        tokens = phrase.split()
        digits = []

        for tok in tokens:
            if tok not in rev:
                return None
            d = rev[tok]
            if d < 0 or d >= base:
                return None
            digits.append(d)

        # positional decode
        out = 0
        for d in digits:
            out = out * base + d
        return out

    def _digits_for_tokens(self, tokens):
        """
        Helper: map list of tokens to digit list using our symbol_map.
        Returns list[int] or None if unknown token encountered.
        """
        if not hasattr(self, "symbol_map"):
            return None
        rev = {v: k for k, v in self.symbol_map.items()}
        digits = []
        for t in tokens:
            if t not in rev:
                return None
            digits.append(rev[t])
        return digits

    # ------------------------------------------------------
    # HELP REQUEST
    # ------------------------------------------------------
    def maybe_request_numeric_help(self, phrase, true_value):
        if not hasattr(self, "api") or self.api is None:
            return
        if not hasattr(self, "population"):
            return

        # choose helper weighted by trust
        peers = [a for a in self.population if a.id != self.id]
        if not peers:
            return

        weights = []
        for p in peers:
            ch = getattr(self, "trust_channels", {}).get(p.id, {})
            trust = (
                0.4 * ch.get("affinity", 0.0)
                + 0.3 * ch.get("reliability", 0.0)
                + 0.3 * ch.get("competence", 0.0)
            )
            weights.append(max(0.1, 1.0 + trust))

        helper = random.choices(peers, weights=weights)[0]
        gen = getattr(self, "current_generation", None)

        line = f"A{self.id} help_numeric helper=A{helper.id} gen={gen} {phrase}?"

        try:
            # use general append_text so we don’t rely on a special helper
            self.api.append_text("/help_required.txt", line + "\n", scope="world")
        except Exception:
            pass

        if hasattr(self, "state"):
            S = self.state
            S["loneliness"] = max(0.0, S["loneliness"] - 0.05)
            S["curiosity"]  = min(1.0, S["curiosity"] + 0.05)

        return helper.id

    # ------------------------------------------------------
    # HELP ANSWERING (GEN-LOCKED + FORWARDING)
    # ------------------------------------------------------
    def maybe_answer_numeric_help(self):
        if not hasattr(self, "api") or self.api is None:
            return
        if not hasattr(self, "population"):
            return

        try:
            raw = (
                self.api.read_help_requests()
                if hasattr(self.api, "read_help_requests")
                else self.api.read_text("/help_required.txt", scope="world")
            )
        except Exception:
            return
        if not raw:
            return

        lines = raw.strip().splitlines()
        lines = lines[-20:]  # last 20 lines only

        seen = set()

        for ln in lines:
            if ln in seen:
                continue
            seen.add(ln)

            parts = ln.split()
            if len(parts) < 5:
                continue

            if parts[1] != "help_numeric":
                continue

            # extract helper + generation
            helper_tag = None
            gen_tag = None
            for p in parts:
                if p.startswith("helper="):
                    helper_tag = p.split("=", 1)[1]
                elif p.startswith("gen="):
                    gen_tag = p.split("=", 1)[1]

            if helper_tag != f"A{self.id}":
                continue
            if gen_tag is None:
                continue

            # generation check
            try:
                req_gen = int(gen_tag)
            except Exception:
                continue

            if req_gen != getattr(self, "current_generation", None):
                continue

            # extract phrase
            try:
                i = parts.index(f"gen={gen_tag}")
            except ValueError:
                continue
            phrase = " ".join(parts[i + 1:]).rstrip("?").strip()
            if not phrase:
                continue

            requester_id = int(parts[0][1:])

            val = self._decode_number_phrase(phrase)
            if val is None:
                # FORWARD to another helper (once)
                peers = [a for a in self.population if a.id != self.id]
                if not peers:
                    return

                weights = []
                for p in peers:
                    ch = getattr(self, "trust_channels", {}).get(p.id, {})
                    trust = (
                        0.4 * ch.get("affinity", 0.0)
                        + 0.3 * ch.get("reliability", 0.0)
                        + 0.3 * ch.get("competence", 0.0)
                    )
                    weights.append(max(0.1, 1.0 + trust))

                nxt = random.choices(peers, weights=weights)[0]
                if nxt.id == self.id:
                    return

                fwd_line = (
                    f"A{requester_id} help_numeric helper=A{nxt.id} "
                    f"gen={req_gen} {phrase}?"
                )
                try:
                    self.api.append_text("/help_required.txt", fwd_line + "\n", scope="world")
                except Exception:
                    pass

                return  # only ONE handled per generation

            # TEACH digits/values
            rev = {v: k for k, v in self.symbol_map.items()}
            tokens = phrase.split()
            digits = []
            for t in tokens:
                if t not in rev:
                    return
                digits.append(rev[t])

            mpairs = ", ".join(f"{d}:{tok}" for d, tok in zip(digits, tokens))

            tline = (
                f"A{self.id} teach_numeric {phrase} "
                f"digits={digits} value={val} map={{ {{ {mpairs} }} }}"
            )

            try:
                if hasattr(self.api, "append_help_response"):
                    self.api.append_help_response(tline)
                else:
                    self.api.append_text("/help_responses.txt", tline + "\n", scope="world")
            except Exception:
                pass

            try:
                if hasattr(self, "adjust_trust"):
                    self.adjust_trust(requester_id, +0.05, channel=4)
            except Exception:
                pass

            return  # one per generation

    # ------------------------------------------------------
    # TEACHING INGESTION (digits + relations)
    # ------------------------------------------------------
    def process_numeric_teaching(self):
        if not hasattr(self, "api") or self.api is None:
            return
        if not hasattr(self, "symbol_map"):
            return

        try:
            txt = (
                self.api.read_help_responses()
                if hasattr(self.api, "read_help_responses")
                else self.api.read_text("/help_responses.txt", scope="world")
            )
        except Exception:
            return

        if not txt:
            return

        lines = txt.strip().splitlines()
        if not lines:
            return

        # new only
        start = self.last_teach_seen
        if start >= len(lines):
            return
        new = lines[start:]
        self.last_teach_seen = len(lines)

        # numeric teaching
        pat_num = re.compile(
            r"^A(?P<teacher>\d+)\s+teach_numeric\s+(?P<phrase>.+?)\s+"
            r"digits=\[(?P<digits>[0-9,\s]+)\]\s+value=(?P<val>\d+)"
        )

        # relation teaching
        pat_rel = re.compile(
            r"^A(?P<teacher>\d+)\s+teach_relation\s+"
            r"(?P<A>.+?)\s+\|\|\s+(?P<B>.+?)\s+rel=(?P<rel>\w+)"
        )

        base = getattr(self.counting, "base", 16)

        for ln in new:
            s = ln.strip()

            # 1) numeric teaching
            m_num = pat_num.match(s)
            if m_num:
                try:
                    teacher = int(m_num.group("teacher"))
                    phrase = m_num.group("phrase").strip()
                    digits = [int(x.strip()) for x in m_num.group("digits").split(",")]
                    tval = int(m_num.group("val"))
                except Exception:
                    continue

                my_val = self._decode_number_phrase(phrase)
                if my_val == tval and my_val is not None:
                    continue

                self._apply_numeric_correction(phrase, digits, tval, teacher, base)
                continue

            # 2) relation teaching
            m_rel = pat_rel.match(s)
            if m_rel:
                try:
                    teacher = int(m_rel.group("teacher"))
                except Exception:
                    continue

                A_str = m_rel.group("A").strip()
                B_str = m_rel.group("B").strip()
                rel = m_rel.group("rel").strip()

                if rel not in ("gt", "lt", "eq"):
                    continue

                if hasattr(self, "update_comparison_operator"):
                    self.update_comparison_operator(A_str.split(), B_str.split(), rel)

                # small semantic shaping on number tokens themselves
                if hasattr(self, "_observe_tokens"):
                    self._observe_tokens((A_str + " " + B_str).split(), gain=0.3)

                # trust bump toward relation teacher
                if hasattr(self, "adjust_trust"):
                    self.adjust_trust(teacher, +0.05, channel=4)

                continue

    # ------------------------------------------------------
    # MODE-2: MAP CORRECTION
    # ------------------------------------------------------
    def _apply_numeric_correction(self, phrase, digits, tval, teacher, base):
        tokens = phrase.split()
        if len(tokens) != len(digits):
            return

        rev = {v: k for k, v in self.symbol_map.items()}

        for tok, d in zip(tokens, digits):
            if d < 0 or d >= base:
                continue

            cur_tok = self.symbol_map.get(d)
            cur_digit = rev.get(tok)

            if cur_tok == tok and cur_digit == d:
                continue

            # assign new
            if cur_digit is None:
                self.symbol_map[d] = tok
                rev[tok] = d

            # swap
            else:
                other_d = cur_digit
                other_tok = self.symbol_map.get(d)

                self.symbol_map[d] = tok
                self.symbol_map[other_d] = other_tok

                rev[tok] = d
                if other_tok is not None:
                    rev[other_tok] = other_d

        # semantic shaping
        if hasattr(self, "_observe_tokens"):
            self._observe_tokens(tokens, gain=0.5)

        # trust
        if hasattr(self, "adjust_trust"):
            self.adjust_trust(teacher, +0.08, channel=4)
            self.adjust_trust(teacher, +0.04, channel=2)

        # emotional
        if hasattr(self, "state"):
            S = self.state
            S["frustration"] = max(0.0, S["frustration"] - 0.05)
            S["confidence"]  = min(1.0, S["confidence"] + 0.05)
            S["curiosity"]   = min(1.0, S["curiosity"] + 0.03)