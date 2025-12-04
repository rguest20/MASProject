from collections import defaultdict

class CommBus:
    """
    Simple message bus. Agents send messages to IDs.
    """
    def __init__(self):
        self.inbox = defaultdict(list)

    def send(self, to_id, payload):
        self.inbox[to_id].append(payload)

    def recv_all(self, agent_id):
        msgs = self.inbox.get(agent_id, [])
        self.inbox[agent_id] = []
        return msgs
