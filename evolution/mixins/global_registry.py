class GlobalTokenRegistry:
    def __init__(self):
        self.numeric_tokens = set()
        self.identity_tokens = set()

    def register_numeric(self, tok):
        self.numeric_tokens.add(tok)

    def is_numeric(self, tok):
        return tok in self.numeric_tokens