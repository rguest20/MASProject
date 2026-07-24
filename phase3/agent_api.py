# file: agent_api.py (corrected)
class AgentAPI:
    # Sandbox I/O is frequent background exploration, not a major survival
    # event.  Whole-unit costs drained the population in roughly ten turns
    # before communication or task learning could have an effect.
    COST_READ  = 0.10
    COST_WRITE = 0.20
    COST_DRAW  = 0.30
    COST_MSG   = 0.10

    def __init__(self, agent, world_fs, private_fs, comm_bus, ledger):
        self.agent  = agent
        self.world  = world_fs
        self.home   = private_fs
        self.bus    = comm_bus
        self.ledger = ledger

    # --------------------------------------------------
    # COST MODEL
    # --------------------------------------------------
    def _enthusiasm_multiplier(self):
        c = self.agent.traits.get("novelty_weight", 0.5)
        w = self.agent.traits.get("cooperation_weight", 0.5)
        enth = 0.5*c + 0.5*w
        return 1.0 - 0.5*enth

    def _charge(self, base, kind):
        cost = base * self._enthusiasm_multiplier()
        self.agent.energy = max(0.0, self.agent.energy - cost)
        self.ledger.note(self.agent.id, kind, cost)
        return cost

    # --------------------------------------------------
    # SAFE TEXT HELPERS
    # --------------------------------------------------
    def _safe_write(self, fs, path, text, append=False):
        """Append or write safely, ensuring parent dirs exist."""
        parent = "/".join(path.split("/")[:-1])
        if parent and parent not in ("", "/") and hasattr(fs, "make_dirs"):
            fs.make_dirs(parent)

        if append:
            fs.append_text(path, text)
        else:
            fs.write_text(path, text)

    # --------------------------------------------------
    # Basic FS I/O
    # --------------------------------------------------
    def list_paths(self, prefix="/", scope="world"):
        if self.agent.energy <= 0.0:
            return []
        fs = self.world if scope == "world" else self.home
        self._charge(self.COST_READ, "read")
        self._report(self.agent, "LIST", {"prefix": prefix, "scope": scope})
        if hasattr(fs, "list_paths"):
            return fs.list_paths(prefix=prefix)
        return []

    def read_text(self, path, scope="world"):
        if self.agent.energy <= 0.0:
            return ""
        fs = self.world if scope == "world" else self.home
        self._charge(self.COST_READ, "read")
        self._report(self.agent, "READ", {"path": path, "scope": scope})
        return fs.read_text(path)

    def write_text(self, path, text, scope="home"):
        if self.agent.energy <= 0.0:
            return
        fs = self.world if scope == "world" else self.home
        self._charge(self.COST_WRITE, "write")
        self._safe_write(fs, path, text)
        self._report(self.agent, "WRITE", {"path": path, "scope": scope})

    def append_text(self, path, text, scope="world"):
        if self.agent.energy <= 0.0:
            return
        fs = self.world if scope == "world" else self.home
        self._charge(self.COST_WRITE, "write")
        self._safe_write(fs, path, text, append=True)
        self._report(self.agent, "WRITE", {"path": path, "scope": scope, "append": True})

    # --------------------------------------------------
    # Canvas
    # --------------------------------------------------
    def draw_pixel(self, x, y, rgb, scope="world"):
        if self.agent.energy <= 0.0:
            return
        fs = self.world if scope == "world" else self.home
        self._charge(self.COST_DRAW, "draw")
        if hasattr(fs, "draw_pixel"):
            fs.draw_pixel(x, y, rgb)
        self._report(self.agent, "DRAW", {"x": x, "y": y, "rgb": rgb, "scope": scope})

    # --------------------------------------------------
    # HELP REQUEST / RESPONSE SYSTEM
    # --------------------------------------------------
    def append_note(self, text, scope="world"):
        """Append to /help_required.txt"""
        if self.agent.energy <= 0.0:
            return
        fs = self.world if scope == "world" else self.home
        self._charge(self.COST_WRITE, "write")
        self._safe_write(fs, "/help_required.txt", text + "\n", append=True)
        self._report(self.agent, "WRITE", {"path": "/help_required.txt", "append": True})

    def read_help_requests(self):
        if self.agent.energy <= 0.0:
            return ""
        self._charge(self.COST_READ, "read")
        self._report(self.agent, "READ", {"path": "/help_required.txt"})
        return self.world.read_text("/help_required.txt")

    def append_help_response(self, text, scope="world"):
        """Teachers write here"""
        if self.agent.energy <= 0.0:
            return
        fs = self.world if scope == "world" else self.home
        self._charge(self.COST_WRITE, "write")
        self._safe_write(fs, "/help_responses.txt", text + "\n", append=True)
        self._report(self.agent, "WRITE", {"path": "/help_responses.txt", "append": True})

    def read_help_responses(self):
        if self.agent.energy <= 0.0:
            return ""
        self._charge(self.COST_READ, "read")
        self._report(self.agent, "READ", {"path": "/help_responses.txt"})
        return self.world.read_text("/help_responses.txt")

    # --------------------------------------------------
    # Messaging
    # --------------------------------------------------
    def send(self, to_id, payload):
        if self.agent.energy <= 0.0:
            return
        self._charge(self.COST_MSG, "message")
        self.bus.send(to_id, {"from": self.agent.id, **payload})
        self._report(self.agent, "MSG", {"to": to_id, "keys": list(payload.keys())})

    def recv_all(self):
        return self.bus.recv_all(self.agent.id)

    # --------------------------------------------------
    def _report(self, agent, kind, meta):
        try:
            if hasattr(agent, "report_action"):
                agent.report_action(kind, meta)
        except:
            pass
