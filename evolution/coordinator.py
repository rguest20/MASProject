# file: evolution/coordinator.py
import math
import random
import re
import json
import secrets
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from agents.agent import Agent
import config as project_config
import evolution.coordinator_settings as coordinator_settings
from evolution.challenge import ChallengeSystem
from agents.cognition.numeric_system import NumericSystem
from evolution.logging import compute_generation_summary, append_generation_to_csv, write_generation_report, _get_log_filenames
from evolution.programs import run_program, safe
from evolution.mixins.global_registry import GlobalTokenRegistry
from evolution.community_lexicon import CommunityLexicon
from evolution.human_dictionary import HumanDictionary
from evolution.community_reading import CommunityReadingRoom
from evolution.coordinator_settings import (
    POP_SIZE,
    ENABLE_DICTIONARY_INJECTION,
    UTTER_EFFECT,
    MATE_POOL_SIZE,
    COMPAT_WEIGHT,
    FITNESS_WEIGHT,
    MEMORY_WEIGHT,
    DIVERSITY_WEIGHT,
    OUTCOME_SCALE,
    MAX_COOP_BONUS,
    MAX_NOVELTY_BONUS,
    ENERGY_MAX,
    IDLE_TAX,
    MIN_PARTICIPATION,
)
from evolution.mixins.coordinator_task_mixin import CoordinatorTaskMixin
from evolution.mixins.coordinator_language_mixin import CoordinatorLanguageMixin
from evolution.community_conversation import CommunityConversation

from evolution.behaviours.orchestration import orchestrate_action


# If your phase3/__init__.py re-exports these, this import works:
from phase3 import SandboxSpec, build_sandbox
from phase3.agent_api import AgentAPI

from evolution.mixins.coordinator_evolution_mixin import CoordinatorEvolutionMixin
from evolution.mixins.coordinator_generation_mixin import CoordinatorGenerationMixin
from evolution.mixins.coordinator_semantic_mixin import CoordinatorSemanticMixin

class Coordinator(CoordinatorGenerationMixin, CoordinatorSemanticMixin, CoordinatorEvolutionMixin, CoordinatorTaskMixin, CoordinatorLanguageMixin):
    def __init__(self, run_dir=None, seed=None, conversation_path=None):
        """Create an isolated, reproducible simulation run.

        Each coordinator owns its reports and sandbox state under a unique
        directory so a new experiment cannot append to or overwrite an older
        one.  ``run_dir`` is available for callers that need to choose the
        destination explicitly; it must not already exist.
        """
        self.random_seed = int(seed) if seed is not None else secrets.randbits(64)
        self.run_dir = self._create_run_dir(run_dir)
        self.report_path = self.run_dir / "generation_report.txt"
        self.csv_path = self.run_dir / "cultural_log.csv"
        self.dialogue_log_path = self.run_dir / "dialogue_log.txt"
        self._write_run_metadata()

        random.seed(self.random_seed)
        np.random.seed(self.random_seed % (2 ** 32))

        self.semantic_alignment_tasks = []
        self.token_registry = GlobalTokenRegistry()
        self.community_lexicon = CommunityLexicon()
        self.human_dictionary = HumanDictionary()

        self.cached_dictionary_words = None
        self.semantic_seeds = {"words": [], "synonyms": [], "antonyms": []}
        self.semantic_seeds = self.build_semantic_seeds()
        self.cached_dictionary_words = self.extract_dictionary_words(limit=300)

        # Core state
        self.generation_index = 0
        self.agents = [Agent(id=i, token_registry=self.token_registry) for i in range(POP_SIZE)]
        for agent in self.agents:
            agent.community_lexicon = self.community_lexicon
            agent.human_dictionary = self.human_dictionary
        self.last_utterances = {}   # agent_id -> utterance
        self.challenge = ChallengeSystem()
        self.action_queue = []

        self.next_task_id = 1
        self.active_tasks = []
        self.referential_memory = []
        self.numeric_memory = []
        self.action_memory = []
        self.human_token_memory = Counter()
        # Tokens introduced by a human prompt, distinct from a human quoting
        # a word that the community already invented.
        self.human_token_origins = {}
        self.human_practice_counts = Counter()
        self.human_practice_recent = deque(maxlen=8)
        self.human_dictionary_link_counts = Counter()
        self.human_sentence_form_votes = Counter()
        self.human_sentence_tokens = Counter()
        self.human_sentence_transitions = defaultdict(Counter)
        # Reading is kept separate from Ryan's transcript.  It supplies a
        # paced context corpus, with a small confidence-gated bridge for
        # words that have been seen in enough independent contexts.
        self.community_reading = CommunityReadingRoom(Path(__file__).resolve().parents[1])
        self.reading_token_memory = Counter()
        self.reading_token_sentences = Counter()
        self.reading_token_context_keys = defaultdict(set)
        self.reading_token_contexts = defaultdict(Counter)
        self.reading_sentence_tokens = Counter()
        self.reading_sentence_transitions = defaultdict(Counter)
        self.reading_sentence_start_tokens = Counter()
        self.reading_sentence_end_tokens = Counter()
        # These are vocabulary candidates, not world facts and not Ryan's
        # transcript.  CommunityConversation only admits them when their
        # reading context overlaps the live topic.
        self.reading_conversation_tokens = Counter()
        self.reading_bridge_scores = {}
        self.reading_context_directions = deque(maxlen=96)
        self.reading_log_path = self.run_dir / "reading_log.txt"
        self.community_learning_state = {
            "quiet_streak": 0,
            "stagnation_streak": 0,
            "wrong_attempt_ema": 0.0,
            "last_read_generation": -999,
            "sentences_read_total": 0,
            "intent_success_total": 0,
        }
        self.community_learning_metrics = {
            "discomfort": 0.0,
            "quiet_streak": 0,
            "stagnation_streak": 0,
            "wrong_attempt_rate": 0.0,
            "reading_sentences": 0,
            "reading_new_tokens": 0,
            "reading_total_sentences": 0,
            "reading_link_agreement": 0.0,
            "reading_bridge_promotions": 0,
            "reading_bridge_vocabulary": 0,
            "intent_proposals": 0,
            "intent_agreement": 0.0,
            "intent_margin": 0.0,
            "intent_target_similarity": 0.0,
            "intent_successes": 0,
        }
        self.community_discomfort = 0.0
        # Reset and filled by run_dialogues each generation.  Keeping this
        # separate from the long dialogue archive makes the current social
        # language pressure visible in the CSV/report.
        self.dialogue_metrics = {}
        # A human-facing transcript lives outside an individual run so it can
        # remain available while a watch-mode simulation continues.  The
        # bridge is inert until a completed ``Ryan: ...`` line appears.
        self.conversation_path = Path(conversation_path or "converse.txt")
        self.community_conversation = CommunityConversation(self.conversation_path)

        self.community_semantic = {
            "vecs": {},          # token -> centroid vector
            "counts": {},        # token -> number of contributors
            "confidence": {},  # token -> 0..1 confidence
            "last_update_gen": 0
        }

        # Build sandbox world + per-agent private FS and APIs
        spec = SandboxSpec(
            root=str(self.run_dir / "sandbox"),
            world_w=64,
            world_h=64,
            seed=self.random_seed % (2 ** 32),
        )
        world, homes, apis, bus, ledger = build_sandbox(self.agents, spec)

        self.world = world
        self.homes = homes
        self.agent_apis = apis
        self.bus = bus
        self.ledger = ledger

        # Attach API, energy, and apply initial semantic seeds to all agents
        for a in self.agents:
            if a.id in self.agent_apis:
                a.attach_api(self.agent_apis[a.id])
            a.energy = ENERGY_MAX

            # Apply global semantic seeds once at initialisation
            try:
                if hasattr(a, "semantic_system") and self.semantic_seeds:
                    a.semantic_system.receive_semantic_seeds(self.semantic_seeds)
            except Exception:
                pass

        # identity grounding for all agents
        for a in self.agents:
            tok = f"A{a.id}"
            a.vocab.add(tok)               # ensure language sees it
            if hasattr(a, "identity_system"):
                a.identity_system.mark_identity_token(tok)
            else:
                a.mark_identity_token(tok)     # fallback bridge

        # Homeostatic control state
        self.archetype_stats = {
            "explorer": 0.0,
            "cooperator": 0.0,
            "habit": 0.0,   # habit learner
            "loner": 0.0,
        }

        # Task weights (will be renormalised by homeostasis)
        self.task_weights = {
            "compare_numbers": 1.0,
            "reconcile_counts": 1.0,
            "agreement_dialogue": 1.0,
        }

        # thresholds for “too low” / “too high”
        self.homeostasis_cfg = {
            "low_frac": 0.10,   # below this → boost
            "high_frac": 0.80,  # above this → damp
            "min_frac_coop": 0.25,
            "max_frac_coop": 0.75,
        }

        self.task_scorers = {
            # already / obvious
            "compare_numbers": self.score_compare_numbers,
            "reconcile_counts": self.score_reconcile_counts,
            "pref_align": self.score_pref_align,

            # coordination pressure
            "agreement_dialogue": self.score_agreement_dialogue,
            "similarity_debate": self.score_similarity_debate,
            "definition_swap": self.score_definition_swap,
            "misunderstanding_detection": self.score_misunderstanding_detection,
        }

        self.completed_tasks = []

    @staticmethod
    def _config_snapshot(module):
        """Return serialisable public constants from a configuration module."""
        return {
            name: value
            for name, value in vars(module).items()
            if name.isupper() and isinstance(value, (str, int, float, bool, type(None)))
        }

    def _create_run_dir(self, requested_dir):
        if requested_dir is not None:
            destination = Path(requested_dir).expanduser().resolve()
            try:
                destination.mkdir(parents=True, exist_ok=False)
            except FileExistsError as exc:
                raise ValueError(
                    f"Run directory already exists: {destination}. "
                    "Choose a new directory to avoid mixing experiment outputs."
                ) from exc
            return destination

        runs_root = Path("runs").resolve()
        runs_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        prefix = f"{timestamp}_seed-{self.random_seed:016x}"

        for suffix in range(1000):
            name = prefix if suffix == 0 else f"{prefix}_{suffix}"
            destination = runs_root / name
            try:
                destination.mkdir()
                return destination
            except FileExistsError:
                continue

        raise RuntimeError("Could not allocate a unique directory for this run.")

    def _write_run_metadata(self):
        metadata = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": self.random_seed,
            "run_directory": str(self.run_dir),
            "artifacts": {
                "generation_report": str(self.report_path),
                "cultural_log": str(self.csv_path),
                "dialogue_log": str(self.dialogue_log_path),
                "sandbox": str(self.run_dir / "sandbox"),
            },
            "config": self._config_snapshot(project_config),
            "coordinator_settings": self._config_snapshot(coordinator_settings),
        }
        with (self.run_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
            file.write("\n")

    def compact_agent_vocabularies(self, max_private_tokens=80):
        """Retain public and recently useful language, not every old token."""
        public = set(self.community_lexicon.numeric_conventions().values())
        public.update(self.community_lexicon.referential_conventions().values())
        public.update(self.community_lexicon.action_conventions().values())
        public.update(getattr(self, "human_token_memory", {}).keys())

        for agent in self.agents:
            protected = set(public)
            protected.update(getattr(agent, "symbol_map", {}).values())
            protected.update(getattr(agent, "referent_lexicon", {}).values())
            protected.update(getattr(agent, "action_lexicon", {}).values())
            protected.update(getattr(agent, "recent_tokens", [])[-40:])
            protected.update(getattr(agent, "reasoning_tokens", {}).values())

            preferences = getattr(agent, "utter_bias", {}).get("symbol_preferences", {})
            candidates = [
                token for token in getattr(agent, "vocab", set())
                if isinstance(token, str) and token.strip() and token not in protected
            ]
            candidates.sort(key=lambda token: (-float(preferences.get(token, 0.0)), token))
            agent.vocab = {
                token for token in protected
                if isinstance(token, str) and token.strip()
            }
            agent.vocab.update(candidates[:max_private_tokens])

    def enqueue_action(self, agent, action):
        self.action_queue.append((agent, action))

    def build_semantic_seeds(self):
        """
        Build a compact semantic seed structure from dictionary.json:
        - word list
        - synonym pairs
        - antonym pairs
        """

        try:
            raw = self.world.read_json("/dictionary.json")
            if not raw or not isinstance(raw, dict):
                return {
                    "words": [],
                    "synonyms": [],
                    "antonyms": []
                }

            words = []
            synonyms = []
            antonyms = []

            for word, entry in raw.items():
                w = word.lower()

                if 2 <= len(w) <= 14 and w.isalpha():
                    words.append(w)

                # SYNONYMS
                for syn in entry.get("SYNONYMS", []):
                    syn = syn.lower()
                    if syn.isalpha() and len(syn) > 1:
                        synonyms.append((w, syn))

                # ANTONYMS
                for ant in entry.get("ANTONYMS", []):
                    ant = ant.lower()
                    if ant.isalpha() and len(ant) > 1:
                        antonyms.append((w, ant))

            # Deduplicate & limit size
            random.shuffle(words)
            words = words[:400]

            return {
                "words": words,
                "synonyms": synonyms,
                "antonyms": antonyms
            }

        except Exception:
            return {
                "words": [],
                "synonyms": [],
                "antonyms": []
            }
        print ("Semantic seeds built:" , len(words), "words;", len(synonyms), "synonyms;", len(antonyms), "antonyms")

    def extract_dictionary_words(self, limit=250):
        if self.cached_dictionary_words is not None:
            return self.cached_dictionary_words[:limit]

        try:
            raw = self.world.read_json("/dictionary.json")
            if not raw or not isinstance(raw, dict):
                self.cached_dictionary_words = []
                return []

            words = []

            for key, entry in raw.items():
                key = key.lower()
                if 2 <= len(key) <= 16:
                    words.append(key)

                for syn in entry.get("SYNONYMS", []):
                    syn = re.sub(r"[^a-z0-9]", "", syn.lower())
                    if 2 <= len(syn) <= 16:
                        words.append(syn)

            random.shuffle(words)
            self.cached_dictionary_words = words
            return words[:limit]

        except Exception:
            self.cached_dictionary_words = []
            return []
