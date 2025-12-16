"""
agents/cognition/pragmatic_system.py
Pragmatic system for managing agent communication and behavior.
"""
class PragmaticSystem:
    """
    Manages the pragmatic aspects of an agent's communication and behavior.
    """
    def __init__(self, owner):
        self.owner = owner
        self.mode_bias = {}       # e.g. ask/assert/mirror
        self.state_bias = {}      # alpha/beta/...
        self.repetition_score = {}

    def observe_utterance(self, utterance):
        """Process an observed utterance pragmatically."""
        return None

    def choose_state(self):
        """Choose the pragmatic state for the next utterance."""
        return None

    def score_novelty(self, utterance):
        """Score the novelty of an utterance."""
        return None
