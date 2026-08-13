# agents/agent_base.py

class AgentBase:
    """
    The minimal, stable base class for all agents.
    Contains only identity, program, API attachment,
    and mixin initialisation glue.

    All cognition, memory, language, emotion, teaching,
    social reasoning, and mutation live in mixins.
    """

    def __init__(self, id=0):
        self.id = id
        self.program = []
        self.energy = 100.0
        self.api = None

        # After super().__init__ in Agent, mixins will run their init units
        # via Agent._post_init(), not here.

    # -----------------------
    # API Attachment
    # -----------------------
    def attach_api(self, api):
        """Bind world API (filesystem, canvas, etc.) to the agent."""
        self.api = api

    # -----------------------
    # Mixin Initialisation Orchestrator
    # -----------------------
    def init_mixins(self):
        """
        Agent calls this to initialise each mixin-specific subsystem.
        Each mixin implements a method starting with `_init_`.
        """
        for name in dir(self):
            if name.startswith("_init_"):
                method = getattr(self, name)
                if callable(method):
                    method()

    # -----------------------
    # Safe logging helpers
    # -----------------------
    def log(self, message, scope="home", path="/log.txt"):
        """
        Optional convenience logger.
        Safe even if no API is attached.
        """
        if not self.api:
            return

        try:
            self.api.append_text(path, f"{message}\n", scope=scope)
        except Exception:
            pass