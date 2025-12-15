# agents/agent.py

from agents.agent_base import AgentBase

# Mixins
from agents.mixins.init_mixin import InitMixin
from agents.mixins.emotion_mixin import EmotionMixin
from agents.mixins.decision_mixin import DecisionMixin
from agents.mixins.action_mixin import ActionMixin
from agents.mixins.language_mixin import LanguageMixin
from agents.mixins.language_mixin_v2 import LanguageMixinV2
from agents.mixins.semantic_mixin import SemanticMixin
from agents.mixins.semantic_stabilisation_mixin import SemanticStabilisationMixin
from agents.mixins.social_mixin import SocialMixin
from agents.mixins.teaching_mixin import TeachingMixin
from agents.mixins.mutation_mixin import MutationMixin
from agents.mixins.sandbox_mixin import SandboxMixin
from agents.mixins.needs_mixin import NeedsMixin
from agents.mixins.util_mixin import UtilMixin
from agents.mixins.task_mixin import TaskMixin
from agents.mixins.task_mixin_v2 import TaskMixinV2
from agents.mixins.reasoning_mixin import ReasoningMixin  # DecisionMixin provides reasoning

class Agent(
    InitMixin,
    UtilMixin,
    EmotionMixin,
    NeedsMixin,
    DecisionMixin,
    ActionMixin,
    ReasoningMixin,
    SemanticMixin,
    SemanticStabilisationMixin,
    LanguageMixinV2,
    SocialMixin,
    TeachingMixin,
    MutationMixin,
    SandboxMixin,
    TaskMixinV2,
    AgentBase,        # Base last
):
    """
    Final unified Agent class.

    All behavioural, cognitive, emotional, linguistic,
    social, and evolutionary machinery lives in mixins.

    AgentBase provides:
      • id
      • energy
      • traits
      • trust_channels
      • counting system
      • basic fitness scoring

    Mixins provide everything else.
    """

    def __init__(self, id, token_registry, api=None):
        super().__init__(id)
        self.token_registry = token_registry
        self._post_init()


    def debug_dump_semantics(self, limit=200):
        """
        Dump the agent's semantic structure for debugging.
        Produces:
          - tokens summary
          - vector norms
          - first dims of each vector
          - link degrees
          - centroid dispersion
        """

        print(f"\n=== SEMANTIC MAP FOR AGENT {self.id} ===")
        print(f"[INIT][A{self.id}] semantic.links id = {id(self.semantic['links'])}")
        vecs = self.semantic.get("vecs", {})
        links = self.semantic.get("links", {})

        toks = list(vecs.keys())
        print(f"Total tokens: {len(toks)}")

        # -----------------------------------------
        # Token summary
        # -----------------------------------------
        print("\n--- TOKENS & VECTOR NORMS ---")
        for i, tok in enumerate(toks[:limit]):
            v = vecs[tok]
            norm = sum(x*x for x in v)**0.5
            preview = ", ".join(f"{x:.3f}" for x in v[:6])
            deg = len(links.get(tok, {}))
            print(f"{tok:12s} | norm={norm:.3f} | deg={deg:3d} | v[:6]=[{preview}]")

        if len(toks) > limit:
            print(f"  (… {len(toks)-limit} more tokens)")

        # -----------------------------------------
        # Vector dispersion across dimensions
        # -----------------------------------------
        print("\n--- DIMENSION VARIANCE ---")
        import numpy as np

        mat = np.array([vecs[t] for t in toks])
        variances = np.var(mat, axis=0)

        for i, v in enumerate(variances[:10]):
            print(f"Dim {i:2d}: var={v:.6f}")

        # -----------------------------------------
        # Link density
        # -----------------------------------------
        print("\n--- LINK DENSITY ---")
        total_links = sum(len(links.get(t, {})) for t in toks)
        avg_links = total_links / max(1, len(toks))
        print(f"Total link edges: {total_links}")
        print(f"Avg links per token: {avg_links:.2f}")

        # -----------------------------------------
        # Optional: cluster preview (distance histogram)
        # -----------------------------------------
        print("\n--- PAIRWISE DISTANCE HISTOGRAM (first 500 pairs) ---")
        import random
        pairs = []
        tok_list = toks.copy()
        random.shuffle(tok_list)
        sample = tok_list[:50]

        def dist(a,b):
            va, vb = vecs[a], vecs[b]
            return sum((x-y)**2 for x,y in zip(va,vb))**0.5

        dists = []
        for i in range(len(sample)):
            for j in range(i+1, len(sample)):
                dists.append(dist(sample[i],sample[j]))

        if not dists:
            print("No sample pairs available.")
        else:
            buckets = [0]*10
            maxd = max(dists)
            for d in dists:
                idx = min(9, int((d/maxd)*10))
                buckets[idx]+=1

            print("Distance buckets:", buckets)