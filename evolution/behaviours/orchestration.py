# evolution/behaviours/orchestration.py

import random


# ============================================================
#  PUBLIC ENTRY POINT
# ============================================================

def orchestrate_action(agent, action, coordinator):
    """
    Dispatch an agent action to the appropriate handler.
    Applies energy cost and emotional updating afterwards.

    Returns:
        success (bool)
    """

    success = False

    # -------------------------
    # Action dispatch
    # -------------------------
    if action == "seek_social":
        success = _social_action(agent, coordinator)

    elif action == "attempt_learning":
        success = _learning_action(agent, coordinator)

    elif action == "attempt_teaching":
        success = _teaching_action(agent, coordinator)

    elif action == "attempt_math_challenge":
        success = _math_challenge(agent, coordinator)

    elif action == "attempt_language_challenge":
        success = _language_challenge(agent, coordinator)

    elif action == "explore_semantic_space":
        success = _semantic_exploration(agent, coordinator)

    elif action == "reorganize_concepts":
        success = _reorganize_semantics(agent)

    # Unknown action → fail safely
    else:
        success = False

    # -------------------------
    # Energy cost
    # -------------------------
    agent.energy -= 0.5
    if agent.energy < 0:
        agent.energy = 0

    # -------------------------
    # Emotional update
    # -------------------------
    try:
        agent.update_emotional_state(action, success)
    except Exception:
        pass

    return success



# ============================================================
#  SOCIAL ACTION
# ============================================================

def _social_action(agent, coordinator):
    """
    Agent attempts to communicate with another agent chosen by trust weighting.
    """
    others = [a for a in coordinator.agents if a.id != agent.id]
    if not others:
        return False

    # trust-weighted partner selection
    weights = []
    for o in others:
        rec = agent.social_memory.get(o.id)
        w = 0.1
        if rec:
            w += max(-0.5, min(1.5, rec["trust_delta"]))
        weights.append(max(0.01, w))

    partner = random.choices(others, weights=weights)[0]

    try:
        utt = agent.produce_utterance()
    except Exception:
        return False

    try:
        coordinator.communicate(agent, partner, utt)
        return True
    except Exception:
        return False



# ============================================================
#  LEARNING ACTION (Semantic Prediction)
# ============================================================

def _learning_action(agent, coordinator):
    teachers = [
        a for a in coordinator.agents
        if a.id != agent.id and a.semantic["vecs"]
    ]
    if not teachers:
        return False

    teacher = random.choice(teachers)
    bundle = teacher.export_semantic_bundle(max_keys=2)
    if not bundle:
        return False

    try:
        agent.attempt_prediction(bundle)
        return True
    except Exception:
        return False



# ============================================================
#  TEACHING ACTION
# ============================================================

def _teaching_action(agent, coordinator):
    students = [a for a in coordinator.agents if a.id != agent.id]
    if not students:
        return False

    student = random.choice(students)
    if not agent.semantic["vecs"]:
        return False

    word = random.choice(list(agent.semantic["vecs"].keys()))

    try:
        agent.teach_student(student, word)
        return True
    except Exception:
        return False



# ============================================================
#  MATH CHALLENGE ACTION
# ============================================================

def _math_challenge(agent, coordinator):
    """
    Agent tries to interpret the challenge number using their counting system.
    """
    challenge_val = coordinator.challenge.challenge_value

    try:
        digits = agent.counting.interpret(challenge_val)
        agent.last_math_attempt = digits
        agent.state_event("curiosity_boost")
        return True
    except Exception:
        return False



# ============================================================
#  LANGUAGE CHALLENGE ACTION
# ============================================================

def _language_challenge(agent, coordinator):
    try:
        utt = agent.produce_utterance()
        agent._last_tokens = utt.split()
        agent.state_event("curiosity_boost")
        return True
    except Exception:
        return False



# ============================================================
#  SEMANTIC EXPLORATION ACTION
# ============================================================

def _semantic_exploration(agent, coordinator):
    """
    Hook for multi-dimensional semantic wandering.
    """
    agent.state_event("curiosity_boost")
    return True



# ============================================================
#  SEMANTIC REORGANISATION ACTION
# ============================================================

def _reorganize_semantics(agent):
    """
    Placeholder: Agent cleans or reshapes their semantic clusters.
    """
    agent.state_event("concept_cleanup")
    return True