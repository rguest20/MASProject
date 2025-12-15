import random

OUTCOME_SCALE          = 0.001

# WORK IN PROGRESS - NOT FULLY INTEGRATED YET
class CoordinatorTaskMixin:
    def _generate_task_id(self):
        tid = f"T{self.next_task_id:04d}"
        self.next_task_id += 1
        return tid

    def load_tasks(self, filename="/tasks.json"):
        """Load tasks from a shared JSON file inside the sandbox."""
        try:
            raw = self.world.read_json(filename)
            if not raw:
                return []
            if isinstance(raw, dict):
                return [raw]
            if isinstance(raw, list):
                return raw
            return []
        except Exception as e:
            print("TASK LOAD ERROR:", e)
            return []

    def generate_describe_concept_task(self):
        if not self.agents:
            return None

        ag = random.choice(self.agents)

        # Pick a token to be described.
        # Prefer tokens with vectors, else random fallback.
        all_tokens = list(getattr(ag.semantic, "vecs", {}).keys())
        if not all_tokens:
            target = "su"
        else:
            target = random.choice(all_tokens)

        return {
            "task_id": self._generate_task_id(),
            "task_type": "describe_concept",
            "assigned_agents": [ag.id],
            "data": {"target_token": target},
            "responses": {}
        }

    def generate_narrative_chain_task(self):
        if not self.agents:
            return None

        ag = random.choice(self.agents)

        # Choose a start token
        toks = list(getattr(ag.semantic, "vecs", {}).keys())
        start = random.choice(toks) if toks else "su"

        return {
            "task_id": self._generate_task_id(),
            "task_type": "narrative_chain",
            "assigned_agents": [ag.id],
            "data": {"start": start},
            "responses": {}
        }

    def generate_prediction_task(self):
        if not self.agents:
            return None

        ag = random.choice(self.agents)

        # produce a random prefix from the agent
        try:
            prefix = ag.produce_utterance()
        except Exception:
            prefix = "su tol"

        return {
            "task_id": self._generate_task_id(),
            "task_type": "prediction_task",
            "assigned_agents": [ag.id],
            "data": {"prefix": prefix},
            "responses": {}
        }

    def generate_similarity_debate_task(self):
        if not self.agents:
            return None

        ag = random.choice(self.agents)

        # grab three candidate tokens
        vecs = getattr(ag.semantic, "vecs", {})
        toks = list(vecs.keys())

        if len(toks) < 3:
            toks = ["su", "tol", "muk"]

        A, B, C = random.sample(toks, 3) if len(toks) >= 3 else ("su", "tol", "muk")

        return {
            "task_id": self._generate_task_id(),
            "task_type": "similarity_debate",
            "assigned_agents": [ag.id],
            "data": {"A": A, "B": B, "C": C},
            "responses": {}
        }

    def generate_role_assignment_task(self):
        if not self.agents:
            return None

        ag = random.choice(self.agents)

        vecs = getattr(ag.semantic, "vecs", {})
        events = list(vecs.keys()) or ["muk"]
        event = random.choice(events)

        return {
            "task_id": self._generate_task_id(),
            "task_type": "role_assignment",
            "assigned_agents": [ag.id],
            "data": {"event": event},
            "responses": {}
        }

    def generate_misunderstanding_detection_task(self):
        if len(self.agents) < 2:
            return None

        a, b = random.sample(self.agents, 2)

        # Ask agent a to produce something ambiguous
        try:
            utt = a.produce_utterance()
        except Exception:
            utt = "su tol rin"

        return {
            "task_id": self._generate_task_id(),
            "task_type": "misunderstanding_detection",
            "assigned_agents": [b.id],   # b interprets a
            "data": {
                "utterance": utt,
                "partner_id": a.id
            },
            "responses": {}
        }

    def generate_definition_swap_task(self):
        if len(self.agents) < 2:
            return None

        a, b = random.sample(self.agents, 2)

        # choose a token to define
        vecs = getattr(a.semantic, "vecs", {})
        toks = list(vecs.keys()) or ["su"]
        tok = random.choice(toks)

        # partner's definition
        try:
            partner_def = a.produce_utterance()
        except Exception:
            partner_def = tok + " su tol"

        return {
            "task_id": self._generate_task_id(),
            "task_type": "definition_swap",
            "assigned_agents": [b.id],
            "data": {
                "token": tok,
                "partner_definition": partner_def
            },
            "responses": {}
        }

    def generate_action_reconstruction_task(self):
        if not self.agents:
            return None

        ag = random.choice(self.agents)

        vecs = getattr(ag.semantic, "vecs", {})
        toks = list(vecs.keys()) or ["su", "tol", "muk"]

        if len(toks) < 2:
            a_tok, b_tok = "su", "tol"
        else:
            a_tok, b_tok = random.sample(toks, 2)

        return {
            "task_id": self._generate_task_id(),
            "task_type": "action_reconstruction",
            "assigned_agents": [ag.id],
            "data": {"from": a_tok, "to": b_tok},
            "responses": {}
        }

    def generate_property_attribution_task(self):
        if not self.agents:
            return None

        ag = random.choice(self.agents)

        vecs = getattr(ag.semantic, "vecs", {})
        toks = list(vecs.keys()) or ["su"]
        target = random.choice(toks)

        return {
            "task_id": self._generate_task_id(),
            "task_type": "property_attribution",
            "assigned_agents": [ag.id],
            "data": {"target_token": target},
            "responses": {}
        }

    def generate_verb_noun_compat_task(self):
        if not self.agents:
            return None

        ag = random.choice(self.agents)

        vecs = getattr(ag.semantic, "vecs", {})
        toks = list(vecs.keys()) or ["muk"]
        verb = random.choice(toks)

        return {
            "task_id": self._generate_task_id(),
            "task_type": "verb_noun_compat",
            "assigned_agents": [ag.id],
            "data": {"verb_token": verb},
            "responses": {}
        }

    def generate_numeric_compare_task(self):
        """
        Produces tasks where A and B may be:
            - single-digit tokens (existing behaviour)
            - OR multi-token base-N numbers.

        Agents must use the *reference agent's* CountingSystem and
        symbol map to interpret the sequences.
        """

        # choose a *reference* agent with stable symbol_map
        ref = random.choice(self.agents)
        smap = getattr(ref, "symbol_map", None)
        if not smap:
            return None

        base = getattr(ref.counting, "base", 16)
        max_n = base ** 2 + random.randint(0, base * 4)   # let them see 2-digit numbers

        # two random integers
        A_val = random.randint(0, max_n)
        B_val = random.randint(0, max_n)

        # convert each to multi-token numeral using the ref's system
        A_tokens = ref.speak_number(A_val).split()
        B_tokens = ref.speak_number(B_val).split()

        A_str = " ".join(A_tokens)
        B_str = " ".join(B_tokens)

        # precompute ground truth in the *same* system (ref)
        if A_val > B_val:
            correct_phrase = A_str
        elif B_val > A_val:
            correct_phrase = B_str
        else:
            correct_phrase = A_str  # tie → A

        return {
            "task_id": self._generate_task_id(),
            "task_type": "compare_numbers",
            "instruction": {
                "base": base,
                "symbol_map": smap,               # 🔴 IMPORTANT LINE
                "format": "compare",
                "description": "Which number is larger?"
            },
            "data": {
                "A": A_str,
                "B": B_str
            },
            # 🔎 Optional: include explicit numeric ground truth for logging
            "ground_truth": {
                "ref_agent": ref.id,
                "A_val": A_val,
                "B_val": B_val,
                "answer_phrase": correct_phrase
            },
            "responses": [],
        }

    def generate_cooperative_compare_task(self):
        """
        Create a cooperative numeric comparison task:
          - pick a reference agent to define the 'ground truth'
          - pick two distinct participants to solve it cooperatively
          - encode A / B using the ref agent's number system
        """
        if not self.agents or len(self.agents) < 2:
            return None

        ref = random.choice(self.agents)
        # simple: sample two random numbers in [0, base^2)
        base = getattr(ref.counting, "base", 8)
        max_val = base * base

        a_val = random.randint(0, max_val - 1)
        b_val = random.randint(0, max_val - 1)
        if a_val == b_val:
            # nudge to avoid constant equality
            b_val = (b_val + 1) % max_val

        A_tokens = ref.counting.interpret(a_val)   # assuming you have this helper
        B_tokens = ref.counting.interpret(b_val)

        A_phrase = " ".join(ref.counting.get_symbol(d) for d in A_tokens)
        B_phrase = " ".join(ref.counting.get_symbol(d) for d in B_tokens)

        # choose two participants
        participants = random.sample(self.agents, 2)
        assigned = [participants[0].id, participants[1].id]

        task = {
            "task_id": self._generate_task_id(),
            "task_type": "compare_numbers",
            "performer": agent.id,   # <-- NEW LINE
            "data": {
                "A": number_phrase_A,
                "B": number_phrase_B,
            },
        }

        # optional: log for debugging
        try:
            line = (
                f"COOP_TASK {task['task_id']} ref=A{ref.id} "
                f"agents={assigned} A='{A_phrase}' B='{B_phrase}' "
                f"a_val={a_val} b_val={b_val}\n"
            )
            self.world.append_text("/coop_tasks.txt", line)
        except Exception:
            pass

        return task

    def generate_agreement_dialogue_task(self):
        # pick 2 distinct agents
        if len(self.agents) < 2:
            return None

        a, b = random.sample(self.agents, 2)
        tid = self._generate_task_id()

        # pick two random utterances or numeric tokens as the “topic”
        # This should be *ambiguous* to encourage negotiation
        try:
            u1 = a.produce_utterance()
            u2 = b.produce_utterance()
        except Exception:
            u1, u2 = "su", "tol"   # fallback

        return {
            "task_id": tid,
            "task_type": "agreement_dialogue",
            "topic": f"{u1} || {u2}",
            "assigned_agents": [a.id, b.id],
        }

    def generate_explain_partner_tasks(self, rounds=10):
        tasks = []
        for _ in range(rounds):
            a, b = random.sample(self.agents, 2)

            # choose a base task to explain (compare_numbers)
            base = self.generate_numeric_compare_task()
            b_answer = b.solve_task(base)

            tasks.append({
                "task_type": "explain_partner",
                "task_id": self.new_task_id(),
                "assigned_agents": [a.id, b.id],
                "partner_answer": b_answer.get("answer", ""),
            })
        return tasks

    def generate_token_compression_tasks(self, rounds=10):
        tasks = []
        for _ in range(rounds):
            ag = random.choice(self.agents)
            # choose random utterance from logs or fabricate
            utt = self.sample_random_utterance() or ag.produce_utterance()

            tasks.append({
                "task_type": "token_compress",
                "task_id": self.new_task_id(),
                "assigned_agents": [ag.id],
                "utterance": utt,
            })
        return tasks

    def generate_preference_alignment_tasks(self, rounds=10):
        tasks = []
        roots = ["tar", "rin", "muk", "vak", "tol", "bel", "zev", "ka", "lo", "su"]

        for _ in range(rounds):
            A, B, P = random.sample(roots, 3)
            a, b = random.sample(self.agents, 2)

            tasks.append({
                "task_type": "pref_align",
                "task_id": self.new_task_id(),
                "assigned_agents": [a.id, b.id],
                "A": A,
                "B": B,
                "pivot": P,
            })
        return tasks

    def generate_reconcile_counts_task(self):
        """
        Habit-learner / pattern-compression task.

        Two agents receive *different descriptions* of a number.
        Their goal is to output the SAME normalized value after decoding.

        Pressure:
        - habit learners: pattern→normal form
        - cooperators: converge on a shared numeric interpretation
        - explorers: less rewarded (stabilising force)
        """
        if len(self.agents) < 2:
            return None

        # choose two different participants
        a, b = random.sample(self.agents, 2)

        # choose reference agent to generate canonical numeric form
        ref = random.choice(self.agents)
        base = getattr(ref.counting, "base", 8)

        # draw a value
        val = random.randint(0, base**2 - 1)

        # each participant gets *its own* encoding of the same number
        def encode(agent, value):
            try:
                toks = agent.speak_number(value).split()
            except Exception:
                toks = ref.speak_number(value).split()
            return " ".join(toks)

        A_view = encode(a, val)
        B_view = encode(b, val)

        tid = self._generate_task_id()

        return {
            "task_id": tid,
            "task_type": "reconcile_counts",
            "value": val,
            "views": {
                a.id: A_view,
                b.id: B_view,
            },
            "assigned_agents": [a.id, b.id],
            "responses": []
        }
    
    def evaluate_cooperative_task(self, task, responses):
        # task was created with key "assigned_agents"
        a_id, b_id = task["assigned_agents"]

        if a_id not in responses or b_id not in responses:
            return

        ra = responses[a_id]
        rb = responses[b_id]

        # Agreement score
        score = 0

        # 1. same relational category
        if ra["relation"] == rb["relation"]:
            score += 1.0

        # 2. same chosen answer token(s)
        if ra["answer"] == rb["answer"]:
            score += 1.0

        # 3. semantic match of shared tokens
        if ra.get("shared_token") and rb.get("shared_token"):
            if ra["shared_token"] == rb["shared_token"]:
                score += 1.0

        # 4. correct numeric comparison
        # Use the phrases we stored under "A" and "B"
        A_full = task["data"]["A"]
        B_full = task["data"]["B"]
        valA = self.agents[a_id]._decode_number_phrase(A_full)
        valB = self.agents[b_id]._decode_number_phrase(B_full)

        correct_rel = (
            REL_GT if valA > valB else
            REL_LT if valB > valA else
            REL_EQ
        )

        if ra["relation"] == correct_rel:
            score += 0.5
        if rb["relation"] == correct_rel:
            score += 0.5

        # 5. reward trust between partners
        self.agents[a_id].adjust_trust(b_id, +0.05 * score, channel=3)
        self.agents[b_id].adjust_trust(a_id, +0.05 * score, channel=3)

        # 6. reward cooperation fitness
        self.agents[a_id].cooperation_bonus += score
        self.agents[b_id].cooperation_bonus += score

    # =======================================================
    # COMMUNITY SEMANTIC MAP
    # ======================================================
    def print_community_semantic_stats(self):
        size = len(self.community_semantic["vecs"])
        print(f"[Community Semantic] Tokens={size}")

    def print_semantic_gap_stats(self, top_k=5):
        total_misaligned = 0
        total_orphans = 0
        samples = []

        for a in self.agents:
            gaps = getattr(a, "semantic_gaps", None)
            if not gaps:
                continue

            mis = gaps.get("misaligned", [])
            orp = gaps.get("orphans", [])

            total_misaligned += len(mis)
            total_orphans += len(orp)

            if mis:
                samples.append((a.id, mis[0]["token"], mis[0]["dist"]))

        print(f"[Semantic Gaps] misaligned={total_misaligned} orphans={total_orphans}")
        if samples:
            samples = sorted(samples, key=lambda x: x[2], reverse=True)[:top_k]
            for aid, tok, dist in samples:
                print(f"  A{aid}: '{tok}' dist={dist:.3f}")

    def community_distance(self, tok, community_map):
        if tok not in community_map["vecs"]:
            return None
        if tok not in self.semantic["vecs"]:
            return None

        a = np.array(self.semantic["vecs"][tok])
        b = np.array(community_map["vecs"][tok])
        return float(np.linalg.norm(a - b))

    # =====================================================
    # Receive gap-driven tasks from agents
    # =====================================================
    def add_semantic_alignment_task(self, task):
        """
        Append a semantic-alignment task proposed by an agent.
        task: {
            "task_id": ...,
            "task_type": "semantic_alignment",
            "token": "...",
            "proposer": agent_id,
            "vector_local": [...],      # proposer's vector
            "community_vector": [...]    # community vector
        }
        """
        self.semantic_alignment_tasks.append(task)

    # =====================================================
    # PHASE D: Fitness shaping for semantic alignment
    # =====================================================
    def evaluate_semantic_alignment_tasks(self):
        """
        Reward agents who:
          - engaged with semantic-alignment tasks, and
          - whose local token semantics are well-aligned with the
            community centroid and the proposer.

        We:
          - reward alignment, not conformity (no hard penalties for
            disagreement, just less bonus)
          - reward proposers if others align with the token they flagged
        """
        if not hasattr(self, "semantic_alignment_tasks"):
            return
        if not getattr(self, "community_semantic", None):
            return

        comm_vecs = self.community_semantic.get("vecs", {})
        proposer_credit = defaultdict(float)

        for task in self.semantic_alignment_tasks:
            tok = task.get("token")
            proposer_id = task.get("proposer")
            if tok is None or proposer_id is None:
                continue

            # community view for this token (may be missing)
            comm_vec = comm_vecs.get(tok)

            # current proposer view (live, not frozen)
            proposer = None
            for ag in self.agents:
                if ag.id == proposer_id:
                    proposer = ag
                    break

            prop_vec = None
            if proposer is not None:
                prop_vec = proposer.semantic["vecs"].get(tok, None)

            # evaluate each agent who actually worked on this token this gen
            for ag in self.agents:
                touched = getattr(ag, "touched_semantic_tokens", set())
                if tok not in touched:
                    continue

                local_vec = ag.semantic["vecs"].get(tok)
                if local_vec is None:
                    continue

                sim_comm = self._cosine(local_vec, comm_vec) if comm_vec is not None else 0.0
                sim_prop = self._cosine(local_vec, prop_vec) if prop_vec is not None else 0.0

                # blend: lean slightly toward community, but still respect proposer
                score = 0.6 * sim_comm + 0.4 * sim_prop

                # exploration-safe: we never *punish* disagreement here,
                # we just give less or no bonus for low/negative scores.
                if score <= 0.0:
                    continue

                # soft cap on contribution per agent per token
                # score is in (0,1]; we scale into ~[0, 1.5]
                raw_reward = min(1.5, score * 1.5)

                # apply to own_fitness and energy gently
                ag.own_fitness += raw_reward
                ag.energy = min(100.0, ag.energy + 0.1 * raw_reward)

                # track semantic alignment success for later analysis
                ag.memory.setdefault("semantic_alignment_success", 0.0)
                ag.memory["semantic_alignment_success"] += raw_reward

                # build proposer credit (if not self-proposed)
                if proposer is not None and proposer.id != ag.id:
                    proposer_credit[proposer.id] += 0.5 * raw_reward

                    # tiny social-memory record to feed into parent selection
                    try:
                        outcome = raw_reward * OUTCOME_SCALE * 200.0
                        ag.remember_interaction(
                            proposer.id,
                            outcome=outcome,
                            gen_index=self.generation_index,
                        )
                        proposer.remember_interaction(
                            ag.id,
                            outcome=outcome * 0.6,
                            gen_index=self.generation_index,
                        )
                    except Exception:
                        pass

        # Apply proposer bonuses (reward "good questions")
        for pid, val in proposer_credit.items():
            for ag in self.agents:
                if ag.id == pid:
                    # proposers benefit, but at lower gain
                    bonus = min(1.0, val)
                    ag.own_fitness += bonus
                    ag.energy = min(100.0, ag.energy + 0.05 * bonus)
                    ag.memory.setdefault("semantic_alignment_proposals", 0.0)
                    ag.memory["semantic_alignment_proposals"] += bonus
                    break