from collections import defaultdict

class RewardLedger:
    """
    Tracks agent behaviour in the sandbox.
    """
    def __init__(self):
        self.stats = defaultdict(lambda: {
            "actions": 0,
            "reads": 0,
            "writes": 0,
            "draws": 0,
            "messages": 0,
            "energy_spent": 0.0,
        })

    def note(self, agent_id, kind, energy_cost=0.0):
        s = self.stats[agent_id]
        s["actions"] += 1
        if kind in ("read", "write", "draw", "message"):
            field = {
                "read": "reads",
                "write": "writes",
                "draw": "draws",
                "message": "messages",
            }[kind]
            s[field] += 1
        s["energy_spent"] += float(energy_cost)

    def reset_gen(self):
        self.stats.clear()
