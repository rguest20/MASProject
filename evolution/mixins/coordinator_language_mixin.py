import math
import random

from agents.cognition.semantic_utils import add, scale, cos_sim
from evolution.coordinator_settings import (
    REFERENTIAL_BONUS,
    PHASE2_LR,
    REWARD_EPS,
    REWARD_TEMP,
    TEACH_PAIRS_PER_PASS,
    TEACH_ACC_TEMP,
    TEACH_TOP_FRACTION,
    TEACH_PROB,
    IMPROVE_ONLY_TEACH,
    OUTCOME_SCALE,
)
from evolution.programs import run_program, mutate_program


class CoordinatorLanguageMixin:
    """
    Language, teaching, and dialogue utilities factored out of the core
    Coordinator.  These methods manipulate agent communication without
    needing to live inside the already large coordinator module.
    """

    def gossip_exchange(self):
        agents = self.agents
        if len(agents) < 2:
            return

        teacher, student = random.sample(agents, 2)
        if random.random() < 0.3:
            if teacher.semantic["vecs"]:
                word = random.choice(list(teacher.semantic["vecs"].keys()))
                teacher.teaching_system.teach_student(student, word)

        for sender in agents:
            rep_list = sender.export_reputation(top_k=5) or []
            conf = float(getattr(sender, "reputation_strength", 0.012))

            num_receivers = random.randint(1, 3)
            receivers = random.sample(agents, min(num_receivers, len(agents)))

            for recv in receivers:
                if recv.id == sender.id:
                    continue

                recv.adjust_trust(
                    target_id=sender.id,
                    amount=min(conf, 0.02),
                    channel=2
                )

                for pid, rep in rep_list:
                    if pid == recv.id or pid == sender.id:
                        continue
                    influence = 0.05 * conf * float(rep)
                    if influence == 0.0:
                        continue
                    recv.adjust_trust(
                        target_id=pid,
                        amount=min(influence, 0.015),
                        channel=3
                    )

                if sender.semantic["vecs"]:
                    shared_word = random.choice(list(sender.semantic["vecs"].keys()))

                    recv.observe_utterance(
                        shared_word,
                        gain_scale=0.25
                    )

    def semantic_teaching_phase(self):
        if not self.agents:
            return

        pairs = min(TEACH_PAIRS_PER_PASS, len(self.agents))
        for _ in range(pairs):
            teacher, student = random.sample(self.agents, 2)

            if teacher.teaching_system.teaching_willingness(student.id) < 0.1 and random.random() < 0.85:
                continue

            bundle = teacher.export_semantic_bundle(max_keys=3)
            if not bundle:
                continue

            pred = student.attempt_prediction(bundle)
            if pred is None:
                continue

            toks = bundle.get("tokens", []) or []
            vecs = teacher.semantic["vecs"]
            gt = None
            try:
                if toks:
                    cols = [vecs[t] for t in toks if t in vecs]
                    if not cols:
                        continue
                    accv = cols[0][:]
                    for v in cols[1:]:
                        accv = add(accv, v)
                    gt = scale(accv, 1.0 / len(cols))
                else:
                    continue
            except Exception:
                continue

            try:
                acc = cos_sim(pred, gt)
            except Exception:
                acc = 0.0

            if TEACH_ACC_TEMP != 1.0 and acc is not None:
                acc = math.copysign(abs(acc) ** TEACH_ACC_TEMP, acc)

            reward = max(-1.0, min(1.0, float(acc)))
            if reward > 0:
                k = 0.10 * reward
            else:
                k = 0.02 * reward

            teacher.apply_teaching_reward(student.id, reward)
            student.apply_learning_reward(teacher.id, reward)

            k = 0.06 * reward
            teacher.update_trust_channels(student.id, reward)
            student.update_trust_channels(teacher.id, reward)

            student.adjust_trust(
                teacher.id,
                amount=max(-0.03, min(0.03, k)),
                channel=4
            )
            teacher.adjust_trust(
                student.id,
                amount=max(-0.02, min(0.02, k * 0.5)),
                channel=3
            )

    def language_feedback(self):
        tots = [a.total_fitness for a in self.agents]
        mmin, mmax = min(tots), max(tots)
        span = (mmax - mmin) if (mmax - mmin) > REWARD_EPS else 1.0

        base_r = {}
        for a in self.agents:
            x01 = (a.total_fitness - mmin) / span
            if REWARD_TEMP != 1.0:
                x01 = x01 ** REWARD_TEMP
            base_r[a.id] = (x01 * 2.0) - 1.0

        def jaccard(a_tokens, b_tokens):
            A, B = set(a_tokens), set(b_tokens)
            if not A or not B:
                return 0.0
            return len(A & B) / len(A | B)

        for listener in self.agents:
            for speaker_id, utt in self.last_utterances.items():
                if listener.id == speaker_id:
                    continue

                r = base_r.get(speaker_id, 0.0)

                speaker = next((x for x in self.agents if x.id == speaker_id), None)
                if speaker is not None:
                    utt_tokens = utt.split()
                    act_tokens = getattr(speaker, "_last_tokens", []) or []
                    r += REFERENTIAL_BONUS * jaccard(utt_tokens, act_tokens)

                r = max(-1.0, min(1.0, r))

                speaker = next((x for x in self.agents if x.id == speaker_id), None)
                if speaker is not None:
                    listener.numeric_system._maybe_learn_numeric_from(speaker)

                listener.learn_from_feedback(utt, reward=r, lr=PHASE2_LR)

        uniq_utts = len({u for u in self.last_utterances.values()})
        if uniq_utts <= 2:
            for a in self.agents:
                try:
                    a.lm.decay(rate=0.99)
                except Exception:
                    pass

    def reward_penalty_phase(self):
        for agent in self.agents:
            net_outcome = sum(v for (_, v) in agent.interaction_memory[-10:]) if hasattr(agent, "interaction_memory") else 0.0
            reward_signal = max(-1.0, min(1.0, net_outcome * 0.1))

            if reward_signal > 0:
                agent.energy += 0.05 * reward_signal
                agent.trust_bias = getattr(agent, "trust_bias", 0.0) + 0.01 * reward_signal
                agent.own_fitness += reward_signal
            elif reward_signal < 0:
                agent.energy -= 0.05 * abs(reward_signal)
                agent.trust_bias = getattr(agent, "trust_bias", 0.0) - 0.01 * abs(reward_signal)
                for pid in agent.social_memory.keys():
                    agent.adjust_trust(pid, amount=-0.005 * abs(reward_signal), channel=4)

        for agent in self.agents:
            agent.energy *= random.uniform(0.96, 0.99)
            agent.energy = min(agent.energy, 100.0)

    def communicate(self, speaker, listener, utterance):
        shared = utterance in listener.utterance_memory["usage_count"]
        coop_avg = (speaker.traits["cooperation_weight"] + listener.traits["cooperation_weight"]) / 2
        success = shared and random.random() < coop_avg

        if success:
            delta_e = 0.5
            speaker.energy = min(100.0, speaker.energy + delta_e)
            listener.energy = min(100.0, listener.energy + delta_e)

            speaker.traits["trust_threshold"] = max(
                0.0, speaker.traits["trust_threshold"] - 0.03
            )
            listener.traits["trust_threshold"] = max(
                0.0, listener.traits["trust_threshold"] - 0.03
            )

            speaker.adjust_trust(listener.id, amount=+0.02, channel=2)
            listener.adjust_trust(speaker.id, amount=+0.02, channel=2)
            speaker.adjust_trust(listener.id, amount=+0.01, channel=3)
            listener.adjust_trust(speaker.id, amount=+0.01, channel=3)
        else:
            delta_e = 0.3
            speaker.energy = max(0.0, speaker.energy - delta_e)
            listener.energy = max(0.0, listener.energy - delta_e)
            speaker.traits["trust_threshold"] = min(
                1.0, speaker.traits["trust_threshold"] + 0.005
            )
            listener.traits["trust_threshold"] = min(
                1.0, listener.traits["trust_threshold"] + 0.005
            )

            speaker.adjust_trust(listener.id, amount=-0.01, channel=4)
            listener.adjust_trust(speaker.id, amount=-0.01, channel=4)

            old = speaker.utterance_memory["associations"].get(utterance, 0.0)
            speaker.utterance_memory["associations"][utterance] = old * 0.9

    @staticmethod
    def crossover_program(prog_a, prog_b):
        try:
            len_a, len_b = len(prog_a), len(prog_b)
            if len_a == 0 or len_b == 0:
                return prog_a[:] if len_a >= len_b else prog_b[:]
            cut_a = random.randrange(len_a)
            cut_b = random.randrange(len_b)
            child = prog_a[:cut_a] + prog_b[cut_b:]
            if len(child) == 0:
                child = (prog_a if random.random() < 0.5 else prog_b)[:]
            return child
        except Exception:
            return prog_a[:]

    def teaching_phase(self):
        ranked = sorted(self.agents, key=lambda a: a.total_fitness, reverse=True)
        top_n = max(1, int(len(self.agents) * TEACH_TOP_FRACTION))
        teachers = ranked[:top_n]

        for teacher in teachers:
            if random.random() > TEACH_PROB:
                continue

            candidates = [a for a in self.agents if a.id != teacher.id]
            if not candidates:
                continue

            weights = []
            for c in candidates:
                will = max(0.0, teacher.teaching_system.teaching_willingness(c.id))
                weights.append(0.05 + will)

            student = random.choices(candidates, weights=weights)[0]

            student.teaching_attempted = True
            teacher.teaching_attempted = True

            if teacher.teaching_system.teaching_willingness(student.id) < teacher.traits.get("trust_threshold", 0.3):
                continue

            new_prog = self.crossover_program(student.program, teacher.program)
            new_prog = mutate_program(new_prog)

            if IMPROVE_ONLY_TEACH:
                old_fit = student.own_fitness
                try:
                    tmp_fit = run_program(new_prog)
                except Exception:
                    tmp_fit = -1e9
                if tmp_fit > old_fit:
                    student.program = new_prog
                    outcome = (tmp_fit - old_fit) * OUTCOME_SCALE
                    student.remember_interaction(teacher.id, outcome=outcome, gen_index=self.generation_index)
                    teacher.remember_interaction(student.id, outcome=outcome * 0.5, gen_index=self.generation_index)
            else:
                student.program = new_prog

            bundle = teacher.export_semantic_bundle(max_keys=1)
            if not bundle:
                continue

            word = bundle["tokens"][0]
            expected_vec = bundle["vecs"][word]

            teach_reward, teach_penalty, sim = teacher.teaching_system.teach_student(
                student, word, current_gen=self.generation_index
            )

            eval_reward = student.teaching_system.evaluate_teaching(
                teacher.id, word, expected_vec
            )

            teacher.apply_teaching_reward(student.id, eval_reward)

            try:
                self.world.append_text(
                    "/notes.txt",
                    f"[TeachPhase] A{teacher.id}->{student.id} "
                    f"trust={teacher.teaching_system.teaching_willingness(student.id):.2f} "
                    f"word={word} sim={sim:.2f} "
                    f"teachR={teach_reward:.2f} evalR={eval_reward:.2f}\n"
                )
            except Exception:
                pass

    def run_language_phase(self):
        self.last_utterances = {}

        for a in self.agents:
            a.challenge_guess = random.randint(0, 4)
            a.challenge_system = self.challenge

        self.challenge.new_challenge()
        self.challenge.assign_liars(self.agents)

        for a in self.agents:
            try:
                utt = a.produce_utterance()
                self.last_utterances[a.id] = utt
                if a.id in self.agent_apis:
                    self.agent_apis[a.id].append_text("/notes.txt", f"A{a.id}: {utt}\n", scope="world")
            except Exception:
                pass

        ids = list(self.last_utterances.keys())
        for _ in range(len(ids)):
            if len(ids) < 2:
                break
            speaker_id, listener_id = random.sample(ids, 2)
            speaker = next(a for a in self.agents if a.id == speaker_id)
            listener = next(a for a in self.agents if a.id == listener_id)
            utt = self.last_utterances[speaker_id]
            self.communicate(speaker, listener, utt)

        for a in self.agents:
            try:
                a.language_world_ingest_step()
            except Exception:
                pass

        self.semantic_teaching_phase()
        self.gossip_exchange()
        self.semantic_teaching_phase()

    def run_dialogues(self, max_pairs_per_gen=30, max_turns_per_pair=2):
        if not hasattr(self, "dialogue_log"):
            self.dialogue_log = []

        proposed_pairs = set()
        for agent in self.agents:
            partner_id = agent.choose_conversation_partner(self.agents)
            if partner_id is None:
                continue

            key = tuple(sorted((agent.id, partner_id)))
            proposed_pairs.add(key)

        if not proposed_pairs:
            return

        proposed_list = list(proposed_pairs)
        random.shuffle(proposed_list)
        chosen_pairs = proposed_list[:max_pairs_per_gen]

        id_to_agent = {a.id: a for a in self.agents}

        for a_id, b_id in chosen_pairs:
            a = id_to_agent.get(a_id)
            b = id_to_agent.get(b_id)
            if a is None or b is None:
                continue

            room_turns = []
            last_utter = None
            last_speaker_id = None

            speaker_order = [a, b] * max_turns_per_pair

            for speaker in speaker_order:
                listener = b if speaker is a else a

                if last_utter is not None and last_speaker_id != listener.id:
                    try:
                        listener.receive_message(last_speaker_id, last_utter)
                    except Exception:
                        pass

                try:
                    if hasattr(speaker, "produce_addressed_utterance"):
                        utter = speaker.produce_addressed_utterance(listener)
                    else:
                        utter = speaker.produce_utterance()
                except Exception:
                    utter = None

                if not utter:
                    utter = ""

                speaker.mark_spoken_turn()
                last_utter = utter
                last_speaker_id = speaker.id

                room_turns.append({
                    "speaker_id": speaker.id,
                    "listener_id": listener.id,
                    "utterance": utter,
                })

            self.dialogue_log.append({
                "generation": getattr(self, "generation_index", None),
                "pair": (a_id, b_id),
                "turns": room_turns[-10:],
            })

        if len(self.dialogue_log) > 5000:
            self.dialogue_log = self.dialogue_log[-5000:]

        with open("dialogue_log.txt", "a") as f:
            for d in self.dialogue_log[-5:]:
                f.write(f"Gen {d['generation']} Pair {d['pair']}\n")
                for t in d["turns"]:
                    f.write(f"  A{t['speaker_id']} → A{t['listener_id']}: {t['utterance']}\n")
                f.write("\n")

    def summarize_dialogues(self, last_n=100):
        if not hasattr(self, "dialogue_log") or not self.dialogue_log:
            return "No dialogues yet."

        recent = self.dialogue_log[-last_n:]
        pairs = {}
        for rec in recent:
            key = tuple(sorted(rec["pair"]))
            pairs.setdefault(key, 0)
            pairs[key] += len(rec["turns"])

        lines = [f"Recent dialogue pairs (last {last_n} records):"]
        for (a, b), count in sorted(pairs.items(), key=lambda x: -x[1])[:15]:
            lines.append(f"  A{a}–A{b}: {count} turns")

        return "\n".join(lines)
