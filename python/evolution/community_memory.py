"""Durable, bounded memory for a community between fresh simulation runs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import DIMS


SCHEMA_VERSION = 1
MAX_TOKENS = 512


class CommunityMemory:
    """Persist public centroids and conventions, never individual agents."""

    def __init__(self, path):
        self.path = Path(path)

    def restore(self, coordinator):
        if not self.path.is_file():
            return {"loaded": False, "tokens": 0, "seeded_agents": 0}
        try:
            with self.path.open("r", encoding="utf-8") as source:
                payload = json.load(source)
        except (OSError, json.JSONDecodeError):
            return {"loaded": False, "tokens": 0, "seeded_agents": 0}
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("dimensions") != DIMS
        ):
            return {"loaded": False, "tokens": 0, "seeded_agents": 0}

        coordinator.community_lexicon.restore_memory_state(payload.get("lexicon", {}))
        semantic = self._clean_semantic(payload.get("semantic", {}))
        coordinator.community_semantic = semantic
        seeded_agents = self._seed_agents(coordinator, semantic)
        return {
            "loaded": True,
            "tokens": len(semantic["vecs"]),
            "seeded_agents": seeded_agents,
        }

    def save(self, coordinator):
        semantic = self._clean_semantic(coordinator.community_semantic)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "implementation": "python",
            "dimensions": DIMS,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "generation": coordinator.generation_index,
            "semantic": semantic,
            "lexicon": coordinator.community_lexicon.memory_state(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as destination:
            json.dump(payload, destination, indent=2, sort_keys=True)
            destination.write("\n")
        os.replace(temporary, self.path)

    @staticmethod
    def _clean_semantic(semantic):
        semantic = semantic if isinstance(semantic, dict) else {}
        vecs = semantic.get("vecs", {})
        counts = semantic.get("counts", {})
        confidence = semantic.get("confidence", {})
        candidates = []
        for token, vector in vecs.items():
            if not isinstance(token, str) or not token.strip() or not isinstance(vector, (list, tuple)):
                continue
            try:
                values = [float(value) for value in vector]
                count = max(0, min(int(counts.get(token, 0)), 1_000_000))
                conf = max(0.0, min(float(confidence.get(token, 0.0)), 1.0))
            except (TypeError, ValueError):
                continue
            if len(values) != DIMS or not np.isfinite(values).all() or count <= 0:
                continue
            candidates.append((token.lower(), values, count, conf))
        candidates.sort(key=lambda item: (item[3], item[2]), reverse=True)
        candidates = candidates[:MAX_TOKENS]
        try:
            last_update_gen = max(0, int(semantic.get("last_update_gen", 0)))
        except (TypeError, ValueError):
            last_update_gen = 0
        return {
            "vecs": {token: vector for token, vector, _, _ in candidates},
            "counts": {token: count for token, _, count, _ in candidates},
            "confidence": {token: conf for token, _, _, conf in candidates},
            "last_update_gen": last_update_gen,
        }

    @staticmethod
    def _seed_agents(coordinator, semantic):
        # A fresh population gets a gentle prior only for established public
        # concepts. It does not inherit private maps, links, fitness, or
        # identity state.
        seeds = [
            (token, np.array(vector, dtype=float), semantic["confidence"].get(token, 0.0))
            for token, vector in semantic["vecs"].items()
            if semantic["confidence"].get(token, 0.0) >= 0.35
        ]
        for agent in coordinator.agents:
            vectors = agent.semantic_system.vectors
            for token, community_vector, confidence in seeds:
                local = np.array(vectors.get(token, agent._rand_vec(DIMS)), dtype=float)
                alpha = 0.18 + 0.32 * confidence
                vectors[token] = ((1.0 - alpha) * local + alpha * community_vector).tolist()
                agent.vocab.add(token)
        return len(seeds)
