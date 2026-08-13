# agents/mixins/util_mixin.py

class UtilMixin:
    """Small shared helpers used by many mixins."""

    def _clamp(self, v, lo=0.0, hi=1.0):
        """Clamp scalar into [lo, hi]."""
        return max(lo, min(hi, v))