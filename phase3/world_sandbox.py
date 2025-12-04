# file: phase3/world_sandbox.py (corrected)
import os, json, shutil
from .persistent_fs import PersistentFS
from .comm import CommBus
from .reward import RewardLedger
from .agent_api import AgentAPI

class SandboxSpec:
    def __init__(self, root="sandbox_root", world_w=32, world_h=32, seed=0):
        self.root = root
        self.world_w = world_w
        self.world_h = world_h
        self.seed = seed


def build_sandbox(agent_list, spec: SandboxSpec):
    root = spec.root
    os.makedirs(root, exist_ok=True)

    # --- WORLD FS ---
    world_path = os.path.join(root, "world")
    os.makedirs(world_path, exist_ok=True)
    world = PersistentFS(world_path)

    # Ensure notes folder exists
    os.makedirs(os.path.join(world_path, "notes"), exist_ok=True)

    # Ensure notes files exist
    # if not os.path.exists(os.path.join(world_path, "notes.txt")):
    world.write_text("/notes.txt", "Shared notes begin here:\n")

    # Dictionary (persist)
    dict_path = os.path.join(world_path, "dictionary.json")
    if not os.path.exists(dict_path):
        world.write_text("/dictionary.json", json.dumps({"hello": "greeting"}))

    # --- NEW: Create help files once ---
    # if not os.path.exists(os.path.join(world_path, "help_required.txt")):
    world.write_text("/help_required.txt", "")

    # if not os.path.exists(os.path.join(world_path, "help_responses.txt")):
    world.write_text("/help_responses.txt", "")

    # --- COMM + LEDGER ---
    bus = CommBus()
    ledger = RewardLedger()

    homes = {}
    apis  = {}

    for a in agent_list:
        home_path = os.path.join(root, f"agent_{a.id}")
        if os.path.exists(home_path):
            shutil.rmtree(home_path)
        os.makedirs(home_path)

        home_fs = PersistentFS(home_path)

        home_fs.write_text("/log.txt", f"agent {a.id} home\n")
        home_fs.write_text("/scratch.txt", "")
        home_fs.write_text("/hints.txt",
            "Try reading /world/notes/welcome.txt\n"
            "Try reading /world/dictionary.json\n"
            "Try writing to /world/notes.txt\n"
            "Try drawing pixels on the world canvas\n"
        )

        # create canvases
        world.canvas = world.create_canvas(spec.world_w, spec.world_h)
        home_fs.canvas = home_fs.create_canvas(16, 16)

        homes[a.id] = home_fs
        apis[a.id]  = AgentAPI(a, world, home_fs, bus, ledger)

    return world, homes, apis, bus, ledger