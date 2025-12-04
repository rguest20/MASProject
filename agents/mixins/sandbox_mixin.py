# agents/mixins/sandbox_mixin.py

import random
import numpy as np
from agents.semantics import rand_vec, add, sub, scale, cos_sim


class SandboxMixin:
    """
    Provides an isolated experimental space inside each agent.
    Used for:
      • hypothesis testing
      • concept recombination
      • private semantic exploration
      • mental simulation of counting or language
      • emotional-resolution actions
      • predictive reasoning

    All sandbox results are intentionally soft:
      They influence emotion, curiosity, and confidence,
      but never directly write to live semantic memory unless
      the agent chooses an explicit "commit" action elsewhere.
    """

    # ------------------------------------------------------------
    # Initial setup — called from Agent._post_init()
    # ------------------------------------------------------------
    def _init_sandbox(self):
        self.sandbox = {
            "scratch": {},        # token -> temp vector
            "last_experiment": "",  # human-readable description
            "last_similarity": 0.0,
            "confidence_boost": 0.0,
        }

    # ------------------------------------------------------------
    # Utility: get or create a vector in sandbox
    # ------------------------------------------------------------
    def _sb_vec(self, tok):
        sb = self.sandbox["scratch"]
        if tok not in sb:
            sb[tok] = rand_vec()
        return sb[tok]

    # ------------------------------------------------------------
    # Experiment A:
    # Hypothesis: "What if concept A were closer to concept B?"
    # ------------------------------------------------------------
    def sandbox_pull_together(self, a, b, strength=0.3):
        """
        Returns similarity increase and emotional feedback.
        Does NOT modify real semantic vectors.
        """
        va = self._sb_vec(a)
        vb = self._sb_vec(b)

        new_a = add(va, scale(sub(vb, va), strength))
        new_b = add(vb, scale(sub(va, vb), strength))

        sim_before = cos_sim(va, vb)
        sim_after = cos_sim(new_a, new_b)

        delta = sim_after - sim_before

        # update scratch only
        self.sandbox["scratch"][a] = new_a
        self.sandbox["scratch"][b] = new_b

        self.sandbox["last_experiment"] = f"pull:{a},{b}"
        self.sandbox["last_similarity"] = delta

        # emotional boost
        if hasattr(self, "state_event"):
            if delta > 0:
                self.state_event("hypothesis_success")
            else:
                self.state_event("hypothesis_conflict")

        return delta

    # ------------------------------------------------------------
    # Experiment B:
    # Blend two concepts into a new emergent idea
    # ------------------------------------------------------------
    def sandbox_blend(self, a, b, name=None, alpha=0.5):
        """
        Creates a new conceptual blend inside sandbox space.
        """
        va = self._sb_vec(a)
        vb = self._sb_vec(b)

        blend_vec = add(scale(va, alpha), scale(vb, 1 - alpha))

        name = name or f"{a}_{b}_blend_{random.randint(0,999)}"
        self.sandbox["scratch"][name] = blend_vec

        self.sandbox["last_experiment"] = f"blend:{a},{b}->{name}"
        self.sandbox["last_similarity"] = cos_sim(va, vb)

        # emotional: blends increase creativity
        if hasattr(self, "state_event"):
            self.state_event("creative_spark")

        return name, blend_vec

    # ------------------------------------------------------------
    # Experiment C:
    # Try predicting a semantic neighbour in sandbox space
    # ------------------------------------------------------------
    def sandbox_guess_neighbor(self, tok):
        """
        Agent attempts to guess what a nearest semantic neighbour
        *might* be, using partial scratch information.
        """
        if not hasattr(self, "semantic"):
            return None

        live_vecs = self.semantic["vecs"]
        sb_vec = self._sb_vec(tok)

        best = None
        best_sim = -1

        for w, v in live_vecs.items():
            sim = cos_sim(sb_vec, v)
            if sim > best_sim and w != tok:
                best = w
                best_sim = sim

        self.sandbox["last_experiment"] = f"guess_neighbor:{tok}->{best}"
        self.sandbox["last_similarity"] = best_sim

        # emotional boost for good guesses
        if hasattr(self, "state_event"):
            if best_sim > 0.6:
                self.state_event("insight")
            else:
                self.state_event("confusion")

        return best

    # ------------------------------------------------------------
    # Experiment D:
    # Play with number patterns in counting system
    # ------------------------------------------------------------
    def sandbox_counting_projection(self, n):
        """
        Try to imagine the representation of a number given current symbols,
        without affecting the real counting system.
        """
        if not hasattr(self, "counting"):
            return None

        # speculative digits
        try:
            seq = self.counting.interpret(n)
        except Exception:
            seq = [random.randint(0, self.counting.base - 1)]

        # experiment: try to create a "mental word" for this representation
        mental = "_".join(str(d) for d in seq)

        self.sandbox["last_experiment"] = f"counting:{n}->{mental}"
        self.sandbox["last_similarity"] = 0.0

        if hasattr(self, "state_event"):
            self.state_event("numerical_play")

        return mental

    # ------------------------------------------------------------
    # Experiment E:
    # Private rehearsal of language
    # ------------------------------------------------------------
    def sandbox_private_utterance(self):
        """
        Agent produces an utterance into the sandbox only.
        Never seen by others. Helps refine language preferences.
        """
        if not hasattr(self, "produce_utterance"):
            return None

        utt = None
        try:
            utt = self.produce_utterance()
        except Exception:
            return None

        cleaned = [t.lower() for t in utt.split()]
        for t in cleaned:
            # slight reinforcement for tokens that 'feel good'
            self._sb_vec(t)

        self.sandbox["last_experiment"] = f"utter:{utt}"
        self.sandbox["last_similarity"] = 0.0

        if hasattr(self, "state_event"):
            self.state_event("self_expression")

        return utt

    # ------------------------------------------------------------
    # Optional: wipe sandbox
    # ------------------------------------------------------------
    def sandbox_clear(self):
        self.sandbox["scratch"].clear()
        self.sandbox["last_experiment"] = ""
        self.sandbox["last_similarity"] = 0.0
        self.sandbox["confidence_boost"] = 0.0


    def _sandbox_policy(self):
        api = getattr(self, "api", None)
        if not api or getattr(self, "energy", 0) <= 0:
            return

        try:
            # read some paths
            if random.random() < 0.40:
                paths = api.list_paths("/", scope="world") or []
                api.append_text("/log.txt", f"seen {len(paths)} world paths\n", scope="home")

            # optional read
            if random.random() < 0.35:
                text = api.read_text("/notes.txt", scope="world")
                if text:
                    api.append_text("/log.txt", f"read notes ({len(text)} chars)\n", scope="home")

            # emit language
            if random.random() < self.traits.get("chattiness", 0.5):
                utt = self.produce_utterance()
                api.append_text("/notes.txt", f"A{self.id}: {utt}\n", scope="world")

            # private scratch
            if random.random() < 0.50:
                api.append_text("/scratch.txt", random.choice(["hi\n", "ok\n", "note\n"]), scope="home")

            # pixel poke
            if random.random() < 0.25:
                W = getattr(api.world, "canvas_w", 32)
                H = getattr(api.world, "canvas_h", 32)
                api.draw_pixel(random.randrange(W), random.randrange(H), (0,0,0), scope="world")

        except Exception:
            pass

    def run_sandbox_step(self):
        self._sandbox_policy()

        api = getattr(self, "api", None)
        if api is None:
            return

        # DSL programs
        if isinstance(self.program, list):
            from evolution.programs import run_program as run_program_dsl
            try:
                run_program_dsl(self.program, api=api)
            except Exception:
                pass

        # callable programs
        elif callable(self.program):
            try:
                self.program(api)
            except Exception:
                pass