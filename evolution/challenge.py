import random
import math

# ----------------------------------------------------
# Numeric concept challenge: agents must represent a quantity
# ----------------------------------------------------

NUM_CHALLENGE_VALUES = 10     # 0..9  (expandable)
LIAR_RATE = 0.10              # 10% liars
CHALLENGE_REWARD = 0.5        # base reward for exact match
PROXIMITY_REWARD_SCALE = 0.25 # scaled reward for near-miss
DIFFICULTY_SCALING = True     # optional: scale proximity reward by difficulty
TRAINING_WHEELS_UNTIL = 30
MIXED_PHASE_UNTIL = 50

CHALLENGE_MEANING = {i: f"num_{i}" for i in range(NUM_CHALLENGE_VALUES)}

def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


class ChallengeSystem:
    """
    Global numeric puzzle for all agents.
    Each generation a number (0–9) is chosen as the hidden target.
    Agents attempt to predict it. Liars misreport their expectations.
    """

    def __init__(self):
        self.challenge_value = random.randint(0, NUM_CHALLENGE_VALUES - 1)
        self.difficulty = 1.0  # tracks how hard current challenge class is

    # ----------------------------------------------------
    # Each generation a new puzzle is created
    # ----------------------------------------------------
    def new_challenge(self):
        self.challenge_value = random.randint(0, NUM_CHALLENGE_VALUES - 1)
        # optionally bias difficulty upward as population skill grows
        self.difficulty = 1.0 + 0.1 * (self.challenge_value / NUM_CHALLENGE_VALUES)

    # ----------------------------------------------------
    # Assign liar flags
    # ----------------------------------------------------
    def assign_liars(self, agents):
        for a in agents:
            a.is_liar = (random.random() < LIAR_RATE)

    # ----------------------------------------------------
    # Compute numeric proximity (distance-based correctness)
    # ----------------------------------------------------
    
    def evaluate_guess(self, agent, generation_index=None):
        """
        Evaluate agent's numeric guess and grant rewards:
        - base reward for correct answer
        - small graded reward for near answers (±1)
        - stability bonus if reuses same symbol->number mapping as previous gens
        """

        true_val = self.challenge_value
        guess = getattr(agent, "challenge_guess", None)
        if guess is None:
            return 0.0

        # Base reward for correctness
        reward = 0.0
        if guess == true_val:
            reward += CHALLENGE_REWARD
        elif abs(guess - true_val) == 1:
            # small partial credit for near-miss
            reward += 0.25 * CHALLENGE_REWARD

        # --- Stability tracking ---
        # agent.symbol_map_history is a short deque or dict tracking (num -> symbol)
        if not hasattr(agent, "symbol_map_history"):
            agent.symbol_map_history = {}

        prev = agent.symbol_map_history.get(true_val)
        current_symbol = agent.symbol_map.get(true_val) if hasattr(agent, "symbol_map") else None

        # Bonus if the mapping hasn't changed since last time we saw this number
        if prev is not None and current_symbol == prev:
            reward += 0.15 * CHALLENGE_REWARD  # stability bonus

        # Update the stored mapping
        if current_symbol is not None:
            agent.symbol_map_history[true_val] = current_symbol

        # Optionally decay reward if agent flips mapping often
        if len(agent.symbol_map_history) > 10:
            # penalise volatility a bit
            reward *= 0.98

        return reward

    # ----------------------------------------------------
    # Expectation ∈ [-1,1] → integer 0..N-1
    # ----------------------------------------------------
    def expectation_to_guess(self, expectation):
        g = int(round((expectation + 1) * (NUM_CHALLENGE_VALUES - 1) / 2))
        return max(0, min(NUM_CHALLENGE_VALUES - 1, g))

    # ----------------------------------------------------
    # For debugging: get string name of the challenge
    # ----------------------------------------------------
    def challenge_name(self):
        return CHALLENGE_MEANING.get(self.challenge_value, str(self.challenge_value))