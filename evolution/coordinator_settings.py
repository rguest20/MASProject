"""
Shared configuration knobs for the Coordinator and its mixins.

Keeping these values in a dedicated module allows the main coordinator
class and the mixins to import them without circular dependencies.
"""

POP_SIZE = 30
ELITE_RATIO = 0.20

ENABLE_DICTIONARY_INJECTION = False

# --- Language ---
UTTER_CHANCE = 1.0
UTTER_EFFECT = 0.15
REWARD_EPS = 1e-9
REWARD_TEMP = 1.0
PHASE2_LR = 0.10
TASK_SEMANTIC_ALIGNMENT = "semantic_alignment"

# --- Social / cultural ---
MATE_POOL_SIZE = 12
COMPAT_WEIGHT = 0.5
FITNESS_WEIGHT = 1.0
MEMORY_WEIGHT = 0.8
DIVERSITY_WEIGHT = 0.2

GOSSIP_PAIRS_PER_GEN = 20
TEACH_PROB = 0.15
TEACH_TOP_FRACTION = 0.20
TEACH_PAIRS_PER_PASS = 16
TEACH_ROUNDS_PER_GEN = 2
TEACH_ACC_TEMP = 1.0
IMPROVE_ONLY_TEACH = True
OUTCOME_SCALE = 0.001
REFERENTIAL_BONUS = 0.35

# --- Safety caps ---
MAX_COOP_BONUS = 1000.0
MAX_NOVELTY_BONUS = 1000.0
MAX_FIT = 1e6

# --- Phase-3 energy / participation ---
ENERGY_MAX = 100.0
IDLE_TAX = 1.0
MIN_PARTICIPATION = 1
