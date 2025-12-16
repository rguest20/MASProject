# agents/mixins/init_mixin.py

import random
from collections import defaultdict, deque
from evolution.counting import CountingSystem
from agents.language import LanguageOrgan
from agents.cognition.numeric_system import NumericSystem
from agents.cognition.epistemic_system import EpistemicSystem
from agents.cognition.identity_system import IdentitySystem
from agents.cognition.semantic_system import SemanticSystem
from agents.cognition.pragmatic_system import PragmaticSystem
from agents.agent_constants import SYLLABLES


class InitMixin:
    """
    Centralised bootstrap initialisation.
    Calls all mixin init methods *including* the reconstructed original
    Agent initialisation (now in _init_core_fields).
    """

    def _post_init(self):
        """
        Called by Agent.__init__()
        Dispatches to all mixin-provided _init_* methods in a safe order.
        """

        # ---- 0. Core original fields from the earlier monolithic Agent ----
        self._init_core_fields()

        # ---- 1. Mixin initialisers (only run if the mixin exists) ----
        init_methods = [
            "_init_agent_identity_semantics",
            "_init_emotion_system",
            "_init_semantic_system",
            "_init_semantic_stabilisation",
            "_init_language_system",
            "_init_social_system",
            "_init_teaching_system",
            "_init_mutation_system",
            "_init_sandbox_system",
            "_init_action_system",
            "_init_needs_system",
            "_init_task_system",
            "_init_reasoning_system",
        ]

        for method in init_methods:
            if hasattr(self, method):
                getattr(self, method)()


    # =====================================================================
    # ORIGINAL AGENT INITIALISATION (100% faithfully preserved)
    # =====================================================================
    def _init_core_fields(self):
        """Reconstructed original Agent initialisation block."""

        # interaction + energy
        self.interaction_memory = []
        self.energy = 100.0
        self.trust_bias = 0.0

        # numeric symbolic mapping
        self.numeric_bias = random.uniform(-1, 1)

        # Cognitive systems
        self.numeric_system = NumericSystem(owner=self)
        self.epistemic_system = EpistemicSystem(owner=self)
        self.identity_system = IdentitySystem(owner=self)
        self.semantic_system = SemanticSystem(owner=self)
        self.pragmatic_system = PragmaticSystem(owner=self)

        self.symbol_map = {}
        self.symbol_map_history = {}
        self.inverse_symbol_map = {}
        self.challenge_guess = 0
        self.is_liar = False
        self.numeric_learning_rate = 0.2

        # trust channels
        self.trust_channels = defaultdict(lambda: {
            "affinity": 0.0,
            "reliability": 0.0,
            "generosity": 0.0,
            "competence": 0.0,
            "consistency": 0.0,
            "collaboration": 0.0,
            "safety": 0.0
        })

        # program
        if not hasattr(self, "program") or self.program is None:
            self.program = []

        # vocabulary
        self.vocab = set(SYLLABLES)
        self.dict_vocab = set()
        self.recent_tokens = []

        # traits
        self.traits = {
            "mutation_rate":        random.uniform(0.05, 0.4),
            "cooperation_weight":   random.uniform(0.0, 1.0),
            "novelty_weight":       random.uniform(0.0, 1.0),
            "stability_weight":     random.uniform(0.0, 1.0),
            "trust_threshold":      random.uniform(0.0, 1.0),
            "chattiness":           random.uniform(0.2, 0.8),
            "patience":             random.uniform(0.2, 0.8),
            "teaching_drive":       random.uniform(0.2, 0.8),
            "curiosity":            random.uniform(0.2, 0.8),
            "expressiveness":       random.uniform(0.2, 0.8),
        }

        # utterance memory
        self.utterance_memory = {
            "associations": {},
            "usage_count": {},
        }

        # language biases
        self.length_bias = random.uniform(0.2, 0.8)
        self.punct_prob = 0.3
        self.utter_bias = {
            "symbol_preferences": {},
            "length_bias": random.uniform(0, 1),
            "repeat_bias": random.uniform(0, 1),
        }

        # social
        self.social_memory = {}
        self.lineage_score = 0.0
        self._last_dict_refresh = None

        # fitness
        self.own_fitness = 0.0
        self.cooperation_bonus = 0.0
        self.novelty_bonus = 0.0
        self.total_fitness = 0.0
        self.esteem = 0.0

        # misc memory
        self.memory = {
            "last_fitness": 0.0,
            "last_fitness_change": 0.0,
            "last_partner": None,
            "cooperation_success": 0.0,
            "novelty_success": 0.0,
        }
        self.memory_influence = random.uniform(0, 1)
        self.memory_decay_rate = random.uniform(0, 1)
        self._scratch = {}

        # dictionary cache
        self._dyn_vocab_upper = set()
        self._dyn_sample = []
        self._dyn_last_refresh_tick = -999999

        # organs
        self.counting = CountingSystem(owner=self)
        self.language = LanguageOrgan(
            owner=self,
            trust_threshold=self.traits["trust_threshold"]
        )

        # motivational needs
        self.needs = {
            "energy": 1.0,
            "social": 0.8,
            "curiosity": 0.7,
            "esteem": 0.5,
            "play": 0.6
        }

        # action list
        self.available_actions = [
            "seek_social",
            "attempt_learning",
            "attempt_teaching",
            "attempt_math_challenge",
            "attempt_language_challenge",
            "explore_semantic_space",
            "reorganize_concepts",
        ]

        self.last_action = None