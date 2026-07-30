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
    Return the report and CSV paths owned by this coordinator run.

    Coordinators created before run-directory support retain the old relative
    paths as a compatibility fallback.
    """
    global _last_block_index, _last_timestamp

    gen = coordinator.generation_index
    block_index = gen // LOG_ROTATE_INTERVAL

    # only assign a new timestamp when we move to a *new* block
    if block_index != _last_block_index:
        _last_block_index = block_index
        _last_timestamp = time.strftime("%Y%m%d_%H%M%S")

    txt_file = getattr(coordinator, "report_path", "generation_report.txt")
    csv_file = getattr(coordinator, "csv_path", "cultural_log.csv")
    return txt_file, csv_file

# -----------------------------
# Utility: cultural signature
# -----------------------------
def effective_numeric_map(agent):
    """Merge promoted public conventions with an agent's private overlay."""
    mapping = {}
    ledger = getattr(agent, "community_lexicon", None)
    if ledger is not None and hasattr(ledger, "numeric_conventions"):
        try:
            mapping.update(ledger.numeric_conventions())
        except Exception:
            pass
    for digit, token in (getattr(agent, "symbol_map", {}) or {}).items():
        mapping.setdefault(digit, token)
    return mapping


def compute_cultural_signature(agent):
    base = getattr(agent.counting, "base", 10)
    mapping = effective_numeric_map(agent)
    symbols = [str(mapping.get(d, "?")) for d in range(base)]
    raw = f"{base}-{'-'.join(symbols)}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def _numeric_alignment_scores(agents):
    """Pairwise agreement scores over shared digit-to-token mappings."""
    scores = []
    for index, left in enumerate(agents):
        left_map = effective_numeric_map(left)
        for right in agents[index + 1:]:
            right_map = effective_numeric_map(right)
            shared = set(left_map) & set(right_map)
            if shared:
                scores.append(sum(left_map[d] == right_map[d] for d in shared) / len(shared))
    return scores


def compute_numeric_alignment(agents):
    """Mean pairwise agreement over shared digit-to-token mappings."""
    scores = _numeric_alignment_scores(agents)
    return sum(scores) / len(scores) if scores else 0.0


def compute_numeric_map_health(agents):
    """Measure coverage and injectivity of maps agents actually use."""
    effective_coverages = []
    overlay_coverages = []
    injectivities = []
    collisions = []
    for agent in agents:
        base = max(1, getattr(getattr(agent, "counting", None), "base", 10))
        overlay = getattr(agent, "symbol_map", {}) or {}
        mapping = effective_numeric_map(agent)
        overlay_coverages.append(sum(bool(overlay.get(d)) for d in range(base)) / base)
        tokens = [mapping[d] for d in range(base) if mapping.get(d)]
        effective_coverages.append(len(tokens) / base)
        injectivities.append(len(set(tokens)) / len(tokens) if tokens else 1.0)
        collisions.append(len(tokens) - len(set(tokens)))
    return (
        statistics.mean(effective_coverages) if effective_coverages else 0.0,
        statistics.mean(overlay_coverages) if overlay_coverages else 0.0,
        statistics.mean(injectivities) if injectivities else 1.0,
        statistics.mean(collisions) if collisions else 0.0,
    )


def compute_grounding_task_metrics(tasks):
    """Summarise attempts, answers, and correct answers by grounding task."""
    groups = {
        "numeric": {"reconcile_counts", "translate_number", "translate_quantity"},
        "referential": {"referential_signal"},
        "action": {"action_signal"},
        "compositional": {"compositional_signal", "compositional_action_signal"},
        "human_dictionary": {"human_dictionary_link"},
    }
    totals = {
        name: {"tasks": 0, "assigned": 0, "attempted": 0, "answered": 0, "correct": 0}
        for name in groups
    }
    for task in tasks:
        task_type = task.get("task_type")
        group = next((name for name, types in groups.items() if task_type in types), None)
        if group is None:
            continue
        evaluation = task.get("evaluation", {}) or {}
        stats = totals[group]
        stats["tasks"] += 1
        stats["assigned"] += int(evaluation.get("assigned", len(task.get("assigned_agents") or [])))
        stats["attempted"] += int(evaluation.get("attempted", len(set(task.get("attempted_by") or []))))
        stats["answered"] += int(evaluation.get("answered", len(task.get("responses") or [])))
        stats["correct"] += int(evaluation.get("correct", 0))

    result = {}
    for name, stats in totals.items():
        prefix = f"{name}_task"
        result.update({
            f"{prefix}s": stats["tasks"],
            f"{prefix}_assigned": stats["assigned"],
            f"{prefix}_attempted": stats["attempted"],
            f"{prefix}_answered": stats["answered"],
            f"{prefix}_correct": stats["correct"],
            f"{prefix}_attempt_rate": stats["attempted"] / stats["assigned"] if stats["assigned"] else 0.0,
            f"{prefix}_answer_rate": stats["answered"] / stats["assigned"] if stats["assigned"] else 0.0,
            f"{prefix}_accuracy": stats["correct"] / stats["answered"] if stats["answered"] else 0.0,
        })
    return result


# -----------------------------
# Summary computation
# -----------------------------
def compute_generation_summary(coordinator):
    agents = coordinator.agents
    gen = coordinator.generation_index

    alignment_scores = _numeric_alignment_scores(agents)
    mean_align = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0
    std_align = statistics.pstdev(alignment_scores) if len(alignment_scores) > 1 else 0.0
    align_div = len({compute_cultural_signature(a) for a in agents})
    numeric_coverage, overlay_coverage, numeric_injectivity, numeric_collisions = compute_numeric_map_health(agents)
    ledger = getattr(coordinator, "community_lexicon", None)
    community_numeric = len(ledger.numeric_conventions()) if ledger is not None else 0
    community_referential = len(ledger.referential_conventions()) if ledger is not None else 0
    community_action = len(ledger.action_conventions()) if ledger is not None else 0
    community_base = ledger.community_base() if ledger is not None else None
    community_grammar = len(ledger.grammar_conventions()) if ledger is not None else 0
    dialogue_metrics = getattr(coordinator, "dialogue_metrics", {}) or {}
    conversation_metrics = getattr(
        getattr(coordinator, "community_conversation", None), "metrics", {}
    ) or {}
    learning_metrics = getattr(coordinator, "community_learning_metrics", {}) or {}

    mean_energy = statistics.mean(a.energy for a in agents)
    mean_fitness = statistics.mean(a.total_fitness for a in agents)
    vocab_mean = statistics.mean(len(getattr(a, "vocab", set())) for a in agents)
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
        "mean_task_fitness": statistics.mean(getattr(a, "task_fitness", 0.0) for a in agents),
        "numeric_map_coverage": numeric_coverage,
        "local_numeric_overlay_coverage": overlay_coverage,
        "numeric_map_injectivity": numeric_injectivity,
        "effective_numeric_map_collisions": numeric_collisions,
        "community_numeric_conventions": community_numeric,
        "community_referential_conventions": community_referential,
        "community_action_conventions": community_action,
        "community_numeric_base": community_base or 0,
        "community_grammar_conventions": community_grammar,
        "vocab_size_mean": vocab_mean,
        "base_diversity": base_div,
        "cluster_diversity": cluster_div,
        "top_clusters": top_clusters,
        "dialogue_pairs": int(dialogue_metrics.get("pairs", 0)),
        "dialogue_turns": int(dialogue_metrics.get("turns", 0)),
        "grounded_dialogue_turns": int(dialogue_metrics.get("grounded_turns", 0)),
        "grounded_dialogue_exchanges": int(dialogue_metrics.get("grounded_exchanges", 0)),
        "grounded_dialogue_successes": int(dialogue_metrics.get("grounded_successes", 0)),
        "teaching_dialogue_exchanges": int(dialogue_metrics.get("teaching_exchanges", 0)),
        "human_token_practice_exchanges": int(dialogue_metrics.get("human_token_practice_exchanges", 0)),
        "free_dialogue_turns": int(dialogue_metrics.get("free_turns", 0)),
        "conversation_pending": int(conversation_metrics.get("pending", 0)),
        "conversation_answers": int(conversation_metrics.get("answers", 0)),
        "conversation_last_agreement": float(conversation_metrics.get("last_agreement", 0.0)),
        "conversation_free_answers": int(conversation_metrics.get("free_answers", 0)),
        "conversation_mode": str(conversation_metrics.get("mode", "introduction")),
        "conversation_active_topic": str(conversation_metrics.get("active_topic", "")),
        "conversation_input_intent": str(conversation_metrics.get("last_input_intent", "")),
        "conversation_reply_intent": str(conversation_metrics.get("last_reply_intent", "")),
        "conversation_mode_agreement": float(conversation_metrics.get("last_mode_agreement", 0.0)),
        "conversation_repeat_penalty": float(conversation_metrics.get("last_repeat_penalty", 0.0)),
        "conversation_working_topic": str(conversation_metrics.get("working_topic", "")),
        "conversation_parked_frames": int(conversation_metrics.get("parked_frames", 0)),
        "conversation_memory_action": str(conversation_metrics.get("last_memory_action", "idle")),
        "conversation_positive_feedback": int(conversation_metrics.get("positive_feedback", 0)),
        "conversation_negative_feedback": int(conversation_metrics.get("negative_feedback", 0)),
        "conversation_last_feedback": str(conversation_metrics.get("last_feedback", "")),
        "conversation_english_bigrams": int(conversation_metrics.get("english_bigrams", 0)),
        "conversation_promoted_bigrams": int(conversation_metrics.get("promoted_bigrams", 0)),
        "conversation_meaning_tokens": int(conversation_metrics.get("meaning_tokens", 0)),
        "conversation_meaning_bigrams": int(conversation_metrics.get("meaning_bigrams", 0)),
        "conversation_world_edges": int(conversation_metrics.get("world_edges", 0)),
        "conversation_world_confirmed_edges": int(conversation_metrics.get("world_confirmed_edges", 0)),
        "community_discomfort": float(learning_metrics.get("discomfort", 0.0)),
        "community_quiet_streak": int(learning_metrics.get("quiet_streak", 0)),
        "community_stagnation_streak": int(learning_metrics.get("stagnation_streak", 0)),
        "community_wrong_attempt_rate": float(learning_metrics.get("wrong_attempt_rate", 0.0)),
        "reading_sentences": int(learning_metrics.get("reading_sentences", 0)),
        "reading_new_tokens": int(learning_metrics.get("reading_new_tokens", 0)),
        "reading_total_sentences": int(learning_metrics.get("reading_total_sentences", 0)),
        "reading_link_agreement": float(learning_metrics.get("reading_link_agreement", 0.0)),
        "reading_bridge_promotions": int(learning_metrics.get("reading_bridge_promotions", 0)),
        "reading_bridge_vocabulary": int(learning_metrics.get("reading_bridge_vocabulary", 0)),
        "conversation_reading_bridge_available": int(conversation_metrics.get("reading_bridge_available", 0)),
        "reading_intent_proposals": int(learning_metrics.get("intent_proposals", 0)),
        "reading_intent_agreement": float(learning_metrics.get("intent_agreement", 0.0)),
        "reading_intent_margin": float(learning_metrics.get("intent_margin", 0.0)),
        "reading_intent_target_similarity": float(learning_metrics.get("intent_target_similarity", 0.0)),
        "reading_intent_successes": int(learning_metrics.get("intent_successes", 0)),
    }
    data.update(compute_grounding_task_metrics(getattr(coordinator, "active_tasks", []) or []))
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
    # Runs already have isolated directories, so never discard earlier
    # generations merely because a long experiment crossed a log block.
    if generation_index == 1 and not os.path.exists(filename):
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
        dialogue_metrics = getattr(coordinator, "dialogue_metrics", {}) or {}
        if dialogue_metrics:
            exchanges = int(dialogue_metrics.get("grounded_exchanges", 0))
            successes = int(dialogue_metrics.get("grounded_successes", 0))
            f.write(
                "   Grounded chat: "
                f"{successes}/{exchanges} understood; "
                f"teaching={int(dialogue_metrics.get('teaching_exchanges', 0))}; "
                f"human-token practice={int(dialogue_metrics.get('human_token_practice_exchanges', 0))}; "
                f"free turns={int(dialogue_metrics.get('free_turns', 0))}\n"
            )
        learning_metrics = getattr(coordinator, "community_learning_metrics", {}) or {}
        f.write(
            "   Learning loop: "
            f"discomfort={float(learning_metrics.get('discomfort', 0.0)):.2f}; "
            f"quiet={int(learning_metrics.get('quiet_streak', 0))}; "
            f"stagnation={int(learning_metrics.get('stagnation_streak', 0))}; "
            f"wrong-rate={float(learning_metrics.get('wrong_attempt_rate', 0.0)):.2f}; "
            f"read={int(learning_metrics.get('reading_sentences', 0))} "
            f"(+{int(learning_metrics.get('reading_new_tokens', 0))} tokens, "
            f"total={int(learning_metrics.get('reading_total_sentences', 0))}); "
            f"links={float(learning_metrics.get('reading_link_agreement', 0.0)):.2f}; "
            f"bridge=+{int(learning_metrics.get('reading_bridge_promotions', 0))}"
            f"/{int(learning_metrics.get('reading_bridge_vocabulary', 0))}; "
            f"intent={int(learning_metrics.get('intent_successes', 0))}"
            f"/{int(learning_metrics.get('intent_proposals', 0))} "
            f"@{float(learning_metrics.get('intent_agreement', 0.0)):.2f} "
            f"margin={float(learning_metrics.get('intent_margin', 0.0)):.2f}\n"
        )
        conversation_metrics = getattr(
            getattr(coordinator, "community_conversation", None), "metrics", {}
        ) or {}
        if conversation_metrics.get("pending") or conversation_metrics.get("answers"):
            f.write(
                "   Conversation bridge: "
                f"status={conversation_metrics.get('last_status', 'idle')}; "
                f"pending={int(conversation_metrics.get('pending', 0))}; "
                f"answers={int(conversation_metrics.get('answers', 0))}; "
                f"free={int(conversation_metrics.get('free_answers', 0))}; "
                f"mode={conversation_metrics.get('mode', 'introduction')}; "
                f"topic={conversation_metrics.get('active_topic', '-') or '-'}; "
                f"intent={conversation_metrics.get('last_input_intent', '-') or '-'}"
                f"->{conversation_metrics.get('last_reply_intent', '-') or '-'}; "
                f"mode-agreement={float(conversation_metrics.get('last_mode_agreement', 0.0)):.2f}; "
                f"repeat-penalty={float(conversation_metrics.get('last_repeat_penalty', 0.0)):.2f}; "
                f"working={conversation_metrics.get('working_topic', '-') or '-'}; "
                f"parked={int(conversation_metrics.get('parked_frames', 0))}; "
                f"memory={conversation_metrics.get('last_memory_action', 'idle')}; "
                f"feedback=+{int(conversation_metrics.get('positive_feedback', 0))}"
                f"/-{int(conversation_metrics.get('negative_feedback', 0))} "
                f"({conversation_metrics.get('last_feedback', '-') or '-'}); "
                f"bigrams={int(conversation_metrics.get('promoted_bigrams', 0))}"
                f"/{int(conversation_metrics.get('english_bigrams', 0))}; "
                f"meaning={int(conversation_metrics.get('meaning_tokens', 0))}"
                f"t/{int(conversation_metrics.get('meaning_bigrams', 0))}b; "
                f"world={int(conversation_metrics.get('world_confirmed_edges', 0))}"
                f"/{int(conversation_metrics.get('world_edges', 0))}; "
                f"agreement={float(conversation_metrics.get('last_agreement', 0.0)):.2f}\n"
            )

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
        ledger = getattr(coordinator, "community_lexicon", None)
        if ledger is not None:
            f.write(
                "\nPublic Conventions: "
                f"numeric={ledger.numeric_conventions()} "
                f"referential={ledger.referential_conventions()} "
                f"actions={ledger.action_conventions()} "
                f"base={ledger.community_base()} "
                f"grammar={ledger.grammar_conventions()}\n"
            )

        f.write("\nEffective Numeric Symbol Maps (sample of 3 agents):\n")
        sample_agents = random.sample(coordinator.agents, min(3, len(coordinator.agents)))
        for ag in sample_agents:
            sm = effective_numeric_map(ag)
            if sm:
                pairs = list(sm.items())
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

                evaluation = task.get("evaluation", {}) or {}
                if evaluation:
                    f.write(
                        "  Outcome: "
                        f"assigned={evaluation.get('assigned', 0)} "
                        f"attempted={evaluation.get('attempted', 0)} "
                        f"answered={evaluation.get('answered', 0)} "
                        f"correct={evaluation.get('correct', 0)}\n"
                    )

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
                elif ttype in {"reconcile_counts", "translate_number", "translate_quantity"}:
                    target = task.get("value")
                    f.write(
                        f"  target={target} base={task.get('numeric_base')} "
                        f"views={task.get('views', {})}\n"
                    )

                    for r in responses:
                        result = r.get("normalized")
                        outcome = "correct" if result == target else "wrong"
                        f.write(
                            f"  Agent {r['agent_id']}: result={result} "
                            f"confidence={r.get('confidence', 0.0):.2f} ({outcome})\n"
                        )

                elif ttype == "referential_signal":
                    task_data = task.get("data", {}) or {}
                    target = task_data.get("referent")
                    f.write(f"  signal='{task_data.get('signal', '')}' target={target}\n")
                    for r in responses:
                        result = r.get("referent")
                        outcome = "correct" if result == target else "wrong"
                        f.write(
                            f"  Agent {r['agent_id']}: referent={result!r} "
                            f"confidence={r.get('confidence', 0.0):.2f} ({outcome})\n"
                        )

                elif ttype == "action_signal":
                    task_data = task.get("data", {}) or {}
                    target = task_data.get("action")
                    f.write(f"  signal='{task_data.get('signal', '')}' target={target}\n")
                    for r in responses:
                        outcome = "correct" if r.get("action") == target else "wrong"
                        f.write(
                            f"  Agent {r['agent_id']}: action={r.get('action')!r} "
                            f"confidence={r.get('confidence', 0.0):.2f} ({outcome})\n"
                        )

                elif ttype == "compositional_signal":
                    task_data = task.get("data", {}) or {}
                    f.write(
                        f"  signal='{task_data.get('signal', '')}' "
                        f"target=({task_data.get('referent')}, {task_data.get('value')})\n"
                    )
                    for r in responses:
                        outcome = (
                            "correct"
                            if r.get("referent") == task_data.get("referent")
                            and r.get("value") == task_data.get("value")
                            else "wrong"
                        )
                        f.write(
                            f"  Agent {r['agent_id']}: referent={r.get('referent')!r} "
                            f"value={r.get('value')} order={r.get('order')} ({outcome})\n"
                        )

                elif ttype == "compositional_action_signal":
                    task_data = task.get("data", {}) or {}
                    f.write(
                        f"  signal='{task_data.get('signal', '')}' target="
                        f"({task_data.get('referent')}, {task_data.get('action')}, {task_data.get('value')})\n"
                    )
                    for r in responses:
                        outcome = (
                            "correct"
                            if r.get("referent") == task_data.get("referent")
                            and r.get("action") == task_data.get("action")
                            and r.get("value") == task_data.get("value")
                            else "wrong"
                        )
                        f.write(
                            f"  Agent {r['agent_id']}: referent={r.get('referent')!r} "
                            f"action={r.get('action')!r} value={r.get('value')} "
                            f"order={r.get('order')} ({outcome})\n"
                        )

                elif ttype == "human_dictionary_link":
                    data = task.get("data", {}) or {}
                    f.write(
                        f"  word={data.get('word')!r} relation={data.get('relation')} "
                        f"options={data.get('options', [])}\n"
                    )
                    accepted = set(data.get("accepted", []) or [])
                    for r in responses:
                        outcome = "correct" if r.get("related") in accepted else "wrong"
                        f.write(
                            f"  Agent {r['agent_id']}: related={r.get('related')!r} "
                            f"confidence={r.get('confidence', 0.0):.2f} ({outcome})\n"
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
