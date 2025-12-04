import random, hashlib
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from .world_fs import VirtualFS

@dataclass
class TaskSpec:
    seed: int
    n_fragments: int = 5
    codeword: str = "LYRABRIDGE"
    canvas_runes: int = 3

def _checksum(frag):
    s = f"{frag['frag']}/{frag['value']}/{frag['salt']}".encode()
    return hashlib.sha256(s).hexdigest()[:12]

def build_tasks_sandbox(spec: TaskSpec, fs: VirtualFS):
    rng = random.Random(spec.seed)
    fragments = []

    # scatter JSON fragments
    for i in range(spec.n_fragments):
        salt = rng.randrange(1, 10_000_000)
        value = spec.codeword[i % len(spec.codeword)]
        frag = {"frag": i, "value": value, "salt": salt}
        proof = _checksum(frag)
        frag["proof"] = proof
        path = f"/world/frag_{i}.json"
        fs.write_text(path, str(frag))
        fragments.append((path, proof))

    # notes
    fs.write_text("/world/notes/guide.txt",
        "Fragments are in /world/*.json. Reconstruct the codeword.\n"
    )

    return {
        "fragments": fragments,
        "codeword": spec.codeword,
    }
