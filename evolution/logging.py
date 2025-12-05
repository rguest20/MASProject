# evolution/logging.py
import statistics
import random
from collections import defaultdict, Counter
import hashlib
import csv
import os
import time
import math

LOG_ROTATE_INTERVAL = 100  # generations
_last_block_index = None
_last_timestamp = None

def _get_log_filenames(coordinator):
    """
    Returns appropriate log file names based on generation index.
    Groups generations into blocks (e.g., 0000–0099, 0100–0199, etc.)
    so multiple generations share the same pair of log files.
    """
    global _last_block_index, _last_timestamp

    gen = coordinator.generation_index
    block_index = gen // LOG_ROTATE_INTERVAL

    # only assign a new timestamp when we move to a *new* block
    if block_index != _last_block_index:
        _last_block_index = block_index
        _last_timestamp = time.strftime("%Y%m%d_%H%M%S")

    # start_block = block_index * LOG_ROTATE_INTERVAL
    # end_block = start_block + LOG_ROTATE_INTERVAL - 1
    # base_name = f"{start_block:04d}-{end_block:04d}_{_last_timestamp}"
    txt_file = "generation_report.txt"
    csv_file = "cultural_log.csv"
    return txt_file, csv_file

# -----------------------------
# Utility: cultural signature
# -----------------------------
def compute_cultural_signature(agent):
    vocab = []
    # if language organ is attached and has dict_vocab
    lang = getattr(agent, "language", None)
    if lang is not None and hasattr(lang, "dict_vocab"):
        vocab = sorted(list(lang.dict_vocab))[:10]

    raw = f"{getattr(agent.counting, 'base', 10)}-{'-'.join(vocab)}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


# -----------------------------
# Summary computation
# -----------------------------
def compute_generation_summary(coordinator):
    agents = coordinator.agents
    gen = coordinator.generation_index

    aligns = [getattr(a, "cultural_alignment", 0.0) for a in agents]
    mean_align = sum(aligns) / len(aligns) if aligns else 0.0
    std_align = statistics.pstdev(aligns) if len(aligns) > 1 else 0.0
    align_div = len(set(round(x, 2) for x in aligns))

    mean_energy = statistics.mean(a.energy for a in agents)
    mean_fitness = statistics.mean(a.total_fitness for a in agents)
    vocab_mean = statistics.mean(
        len(getattr(getattr(a, "language", None), "dict_vocab", []))
        for a in agents
    )
    bases = [a.counting.base for a in agents if hasattr(a, "counting")]
    base_div = len(set(bases))

    sigs = [compute_cultural_signature(a) for a in agents]
    counts = Counter(sigs)
    top_clusters = ",".join([f"{k}:{v}" for k, v in counts.most_common(3)])
    cluster_div = len(counts)

     # -------------------------------
    # Motivational State Summary
    # -------------------------------
    try:
        if hasattr(agents[0], "needs"):
            need_keys = list(agents[0].needs.keys())
            for k in need_keys:
                vals = [a.needs[k] for a in agents]
                mean = statistics.mean(vals)
                std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                # add mean + std for CSV logging
                field_mean = f"need_{k}_mean"
                field_std = f"need_{k}_std"
                # directly add into data dict to return
                # (using dict.update avoids overwriting)
                data_part = {
                    field_mean: round(mean, 3),
                    field_std: round(std, 3)
                }
                # we’ll merge this into the final return dictionary later
                globals().setdefault("_last_need_data", {}).update(data_part)
    except Exception as e:
        print(f"[Motivation log error: {e}]")

    # merge need data into return
    data = {
        "generation": gen,
        "mean_alignment": mean_align,
        "alignment_std": std_align,
        "alignment_diversity": align_div,
        "mean_energy": mean_energy,
        "mean_fitness": mean_fitness,
        "vocab_size_mean": vocab_mean,
        "base_diversity": base_div,
        "cluster_diversity": cluster_div,
        "top_clusters": top_clusters,
    }
    if "_last_need_data" in globals():
        data.update(_last_need_data)
    return data


def append_generation_to_csv(data, filename="cultural_log.csv"):
    file_exists = os.path.isfile(filename)
    headers = list(data.keys())
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

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


# -----------------------------
# Full textual generation log
# -----------------------------
def write_generation_report(coordinator, filename="generation_report.txt", generation_index=0):
    """
    Full text version of the previous print_stats + rich_generation_log.
    """
    # ensure file at filename is empty at start of new block
    if (generation_index -1) % LOG_ROTATE_INTERVAL == 0:
        with open(filename, "w") as f:
            f.write(f"Generation Report Log\nStarted at {time.ctime()}\n")

    if (coordinator.generation_index % LOG_ROTATE_INTERVAL) == 0:
        with open(filename, "a") as f:
            f.write(f"\n=== New Log Block Started at {time.ctime()} ===\n")
    with open(filename, "a") as f:
        f.write(f"\n=== Generation {coordinator.generation_index} ===\n")

        # Language summary
        counts = defaultdict(int)
        for utt in coordinator.last_utterances.values():
            counts[utt] += 1

        f.write("\nLanguage Summary:\n")
        if counts:
            top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for utt, c in top:
                f.write(f"   '{utt}' : {c} uses\n")
        else:
            f.write("   (no utterances)\n")
        f.write(f"   Unique utterances: {len(counts)}\n")
        f.write(f"Challenge this gen: {coordinator.challenge.challenge_value} "
                f"({coordinator.challenge.challenge_name()})\n")

        # Tail of notes
        try:
            notes = coordinator.world.read_text("/notes.txt") or coordinator.world.read_text("/world/notes.txt")
            if notes:
                tail = "\n".join(notes.strip().splitlines()[-5:])
                f.write("\nShared notes tail:\n  " + tail.replace("\n", "\n  ") + "\n")
        except Exception:
            pass

        # Traits summary
        f.write("\n=== Diagnostic Summary ===\n")
        trait_keys = [
            "mutation_rate", "cooperation_weight", "novelty_weight",
            "stability_weight", "trust_threshold",
        ]
        f.write("\nTrait Summary:\n")
        for key in trait_keys:
            vals = [a.traits[key] for a in coordinator.agents if key in a.traits]
            if not vals:
                continue
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            f.write(f"  {key:<17} mean={mean:.3f} std={std:.3f} "
                    f"min={min(vals):.3f} max={max(vals):.3f}\n")

        mem_inf = [a.memory_influence for a in coordinator.agents]
        mem_dec = [a.memory_decay_rate for a in coordinator.agents]
        if mem_inf or mem_dec:
            f.write("\nMemory Summary:\n")
            if mem_inf:
                f.write(f"  memory_influence   mean={statistics.mean(mem_inf):.3f}\n")
            if mem_dec:
                f.write(f"  memory_decay_rate  mean={statistics.mean(mem_dec):.3f}\n")

        social_keys = [
            "chattiness", "patience", "teaching_drive", "curiosity", "expressiveness"
        ]
        f.write("\nSocial / Linguistic Traits:\n")
        for key in social_keys:
            vals = [a.traits[key] for a in coordinator.agents if key in a.traits]
            if not vals:
                continue
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            f.write(f"  {key:<17} mean={mean:.3f} std={std:.3f} "
                    f"min={min(vals):.3f} max={max(vals):.3f}\n")

        # Program diversity
        programs_repr = [repr(a.program) for a in coordinator.agents]
        unique = len(set(programs_repr))
        diversity = unique / len(programs_repr) if programs_repr else 0.0
        f.write(f"\nProgram Diversity: {diversity:.2f} "
                f"({unique}/{len(programs_repr)} unique)\n")

        # Archetypes
        explorers = sum(1 for a in coordinator.agents if a.traits["novelty_weight"] > 0.75)
        cooperators = sum(1 for a in coordinator.agents if a.traits["cooperation_weight"] > 0.80)
        loners = sum(1 for a in coordinator.agents if a.traits["trust_threshold"] > 0.70)
        habit_learners = sum(1 for a in coordinator.agents if a.memory_influence > 0.70)
        f.write("\nEmergent Archetypes:\n")
        f.write(f"  Explorers:   {explorers}\n")
        f.write(f"  Cooperators: {cooperators}\n")
        f.write(f"  Loners:      {loners}\n")
        f.write(f"  Habit Learners: {habit_learners}\n")

        # Top agents
        ranked = sorted(coordinator.agents, key=lambda a: a.total_fitness, reverse=True)
        top5 = ranked[:5]
        f.write("\nTop Agents:\n")
        for a in top5:
            f.write(f"  ID {a.id:<2} fit={a.total_fitness:.3f}\n")

        if ranked:
            best = ranked[0]
            f.write("\nBest Agent This Gen:\n")
            f.write(f"  ID {best.id} (fitness={best.total_fitness:.4f})\n")
            for key in trait_keys:
                f.write(f"  {key:<17} = {best.traits[key]:.3f}\n")
            f.write(f"  memory_influence   = {best.memory_influence:.3f}\n")
            f.write(f"  memory_decay_rate  = {best.memory_decay_rate:.3f}\n")

        # Energy summary
        energies = [a.energy for a in coordinator.agents]
        f.write("\nEnergy Summary:\n")
        f.write(f"  mean={statistics.mean(energies):.2f} "
                f"min={min(energies):.2f} max={max(energies):.2f}\n")

        # Symbol maps
        f.write("\nNumeric Symbol Maps (sample of 3 agents):\n")
        sample_agents = random.sample(coordinator.agents, min(3, len(coordinator.agents)))
        for ag in sample_agents:
            if getattr(ag, "symbol_map", None):
                pairs = list(ag.symbol_map.items())
                pairs_str = ", ".join(f"{k}->{v}" for k, v in pairs)
                f.write(f"  A{ag.id}: {pairs_str}\n")

        # Counting system diversity
        bases = [getattr(a.counting, "base", None)
                 for a in coordinator.agents if hasattr(a, "counting")]
        bases = [b for b in bases if b is not None]
        if bases:
            unique_bases = sorted(set(bases))
            f.write(f"\nCounting base diversity: {len(unique_bases)} "
                    f"unique bases -> {unique_bases}\n")

        # Motivational summary
        if hasattr(coordinator.agents[0], "needs"):
            f.write("\nMotivational State:\n")
            need_keys = list(coordinator.agents[0].needs.keys())
            for k in need_keys:
                vals = [a.needs[k] for a in coordinator.agents]
                mean = statistics.mean(vals)
                std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                f.write(f"  {k:<10} mean={mean:.3f} std={std:.3f} "
                        f"min={min(vals):.3f} max={max(vals):.3f}\n")

        # === Task Results ===
        f.write("\n=== Task Results ===\n")

        active_tasks = getattr(coordinator, "active_tasks", []) or []
        if not active_tasks:
            f.write("  (no active tasks)\n")

        else:
            for task in active_tasks:
                ttype = task.get("task_type")
                tid = task.get("task_id")
                data = task.get("data", {})
                inst = task.get("instruction", {})

                f.write(f"\nTask {tid} ({ttype}):\n")

                # ---------------------------------------------------------
                # Collect all agent responses for this task
                # ---------------------------------------------------------
                responses = []
                for ag in coordinator.agents:
                    solved = getattr(ag, "solved_tasks", {})
                    resp = solved.get(tid)
                    if resp:
                        responses.append(resp)

                if not responses:
                    f.write("  No agent responses.\n")
                    continue

                # =========================================================
                # TASK TYPE: compare_numbers
                # =========================================================
                if ttype == "compare_numbers":
                    f.write(f"  data={data}\n")

                    gt = task.get("ground_truth", {})
                    correct = gt.get("answer_phrase", None)

                    if correct is not None:
                        f.write(f"  Correct answer (ref agent) = {correct}\n")

                    for r in responses:
                        ans = r.get("answer")
                        conf = r.get("confidence", 0.0)
                        just = r.get("justification", "")

                        # correctness check
                        if correct is None:
                            outcome = ""
                        elif correct == "equal":
                            outcome = " (equal-case)"
                        elif ans == correct:
                            outcome = " (correct)"
                        else:
                            outcome = " (wrong)"

                        f.write(
                            f"  Agent {r['agent_id']}: answer={ans} "
                            f"conf={conf:.3f}{outcome} "
                            f"justification='{just}'\n"
                        )

                # =========================================================
                # TASK TYPE: reconcile_counts
                # =========================================================
                elif ttype == "reconcile_counts":
                    f.write(f"  utterance='{data.get('utterance', '')}'\n")

                    for r in responses:
                        mode = r.get("mode", "?")
                        result = r.get("result", "")
                        f.write(
                            f"  Agent {r['agent_id']}: mode={mode} result='{result}'\n"
                        )

                # =========================================================
                # TASK TYPE: agreement_dialogue
                # =========================================================
                elif ttype == "agreement_dialogue":
                    topic = task.get("topic", "")
                    f.write(f"  topic='{topic}'\n")

                    for r in responses:
                        prop = r.get("proposal", "")
                        reuse = r.get("reused_partner_tokens", False)
                        f.write(
                            f"  Agent {r['agent_id']}: proposal='{prop}' reused={reuse}\n"
                        )

                # =========================================================
                # TASK TYPE: explain_partner
                # =========================================================
                elif ttype == "explain_partner":
                    partner_answer = task.get("partner_answer", "")
                    f.write(f"  partner_answer='{partner_answer}'\n")

                    for r in responses:
                        expl = r.get("explanation", "")
                        reused = r.get("tokens_reused", [])
                        f.write(
                            f"  Agent {r['agent_id']}: explanation='{expl}' reused={reused}\n"
                        )

                # =========================================================
                # TASK TYPE: semantic_alignment
                # =========================================================
                elif ttype == "semantic_alignment":
                    token = data.get("token")
                    f.write(f"  token='{token}'\n")

                    for r in responses:
                        agree = r.get("agree", False)
                        dist = r.get("distance", 0.0)
                        reward = r.get("reward", 0.0)
                        f.write(
                            f"  Agent {r['agent_id']}: agree={agree} dist={dist:.3f} reward={reward:.3f}\n"
                        )

                # =========================================================
                # TASK TYPE: preference_alignment_dialogue
                # =========================================================
                elif ttype == "pref_align":
                    A = task.get("A")
                    B = task.get("B")
                    P = task.get("pivot")

                    f.write(f"  A={A}, B={B}, pivot={P}\n")

                    for r in responses:
                        choice = r.get("choice")
                        utter = r.get("utterance", "")
                        simA = r.get("simA", 0.0)
                        simB = r.get("simB", 0.0)
                        f.write(
                            f"  Agent {r['agent_id']}: choice={choice} simA={simA:.3f} simB={simB:.3f} utter='{utter}'\n"
                        )

                # =========================================================
                # TASK TYPE: token_compress
                # =========================================================
                elif ttype == "token_compress":
                    utt = task.get("utterance", "")
                    f.write(f"  input='{utt}'\n")

                    for r in responses:
                        mode = r.get("mode")
                        result = r.get("result", "")
                        f.write(
                            f"  Agent {r['agent_id']}: mode={mode} result='{result}'\n"
                        )

                # =========================================================
                # UNKNOWN OR NEW TASK TYPE
                # =========================================================
                else:
                    f.write("  (no specialised logger; dumping raw responses)\n")
                    for r in responses:
                        f.write(f"    {r}\n")