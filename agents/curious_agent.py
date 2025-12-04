from phase3 import AgentAPI

class CuriousAgent:

    def __init__(self, id):
        self.id = id
        self.memory = {"last_fitness": 0.0}
        self.own_fitness = 0.0

    def program(self, api: AgentAPI):
        """
        This is the "callable" used by Phase-3.
        Evolved agents will replace this with whatever genome → code
        you are generating. For now, we give a functional stub.
        """

        # ---- Listen first ----
        heard = api.inbox()
        proofs = []
        for m in heard:
            if (
                m.payload.get("type") == "claim" and
                m.payload.get("what") == "frag"
            ):
                proofs.append(m.payload.get("proof"))

        # ---- Explore FS ----
        root = api.listdir("/world")
        read = 0
        for name in root:
            if name.endswith(".json") and read < 3:
                obj = api.read_json(f"/world/{name}")
                if obj and "proof" in obj:
                    api.claim_frag(obj["frag"], obj["proof"])
                    proofs.append(obj["proof"])
                    read += 1

        # ---- Attempt final codeword if confident ----
        if len(set(proofs)) >= 3:
            api.claim_codeword(
                candidate="LYRABRIDGE",
                evidence=list(set(proofs)),
            )

        # ---- Leave field notes ----
        api.append_note(
            "field.log.txt",
            f"Agent {api.agent_id} observed {len(root)} files, {len(proofs)} proofs"
        )
