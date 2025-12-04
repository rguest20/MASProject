import math
from typing import List
from phase3 import (
    VirtualFS, SandboxSpec, build_sandbox,
    CommBus, AgentAPI, RewardLedger
)

SAFE_FITNESS_CAP = 300.0


class CoordinatorPhase3:

    def __init__(self, seed: int, agents: List):
        self.seed = seed
        self.agents = agents

    def run_generation(self, generation_index: int):
        """
        One full phase-3 loop:
        ✅ Create sandbox (virtual world)
        ✅ Let agents explore & communicate
        ✅ Score their discoveries & communications
        ✅ Apply a safe fitness update
        """

        # -------------------------------
        # 1. Sandbox setup
        # -------------------------------
        fs = VirtualFS(seed=self.seed + generation_index)
        spec = SandboxSpec(seed=self.seed + 31 * generation_index)
        manifest = build_sandbox(spec, fs)

        bus = CommBus(seed=self.seed ^ generation_index)
        ledger = RewardLedger(spec, manifest, fs, bus)

        # -------------------------------
        # 2. Agent execution
        # -------------------------------
        for agent in self.agents:
            api = AgentAPI(
                agent_id=str(agent.id),
                t=generation_index,
                fs=fs,
                bus=bus,
            )
            try:
                agent.program(api)   # call the agent's script
            except Exception:
                pass  # sandbox shielding

        # -------------------------------
        # 3. Scoring (verifiable discoveries)
        # -------------------------------
        deltas = ledger.score_generation(
            agent_ids=[str(a.id) for a in self.agents]
        )

        # -------------------------------
        # 4. Apply safe bounded fitness
        # -------------------------------
        for agent in self.agents:
            delta = float(deltas.get(str(agent.id), 0.0))
            last = float(agent.memory.get("last_fitness", 0.0))

            updated = last + delta

            # Hard anti-exploit protections
            if updated != updated:
                updated = 0.0
            updated = max(-1e4, min(updated, 1e4))
            updated = math.log1p(updated)
            updated = min(updated, SAFE_FITNESS_CAP)

            # store
            agent.memory["last_fitness_change"] = updated - last
            agent.memory["last_fitness"] = updated
            agent.own_fitness = updated
