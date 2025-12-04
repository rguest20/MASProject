# agents/agent_utils.py
import random

def clamp01(x):
    """Clamp a scalar to [0, 1]."""
    return max(0.0, min(1.0, float(x)))

def clamp(x, lo, hi):
    """Clamp to an arbitrary range."""
    return max(lo, min(hi, x))

def _clamp(x, lo, hi):
    """Clamp to an arbitrary range."""
    return max(lo, min(hi, x))

def weighted_choice(items, weights):
    """Safe weighted random choice."""
    if not items:
        return None
    w = [max(1e-9, float(x)) for x in weights]
    return random.choices(items, weights=w)[0]