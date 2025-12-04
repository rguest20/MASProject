# file: evolution/language_v2.py
import json, math, random
from collections import Counter, defaultdict

# Reasonable defaults
DEFAULT_BASE_TOKENS = [
    "hi","ok","note","yes","no","maybe","ping","hello","share","read","write"
]

def _tokenize(text):
    out = []
    w = []
    for ch in text.lower():
        if "a" <= ch <= "z":
            w.append(ch)
        elif ch.isdigit():
            w.append(ch)
        else:
            if w:
                tok = "".join(w)
                if 1 <= len(tok) <= 16:
                    out.append(tok)
                w = []
    if w:
        tok = "".join(w)
        if 1 <= len(tok) <= 16:
            out.append(tok)
    return out


class TinyLM:
    """
    Minimal learnable language model:
    - Unigram counts
    - Bigram transition table
    - Temperature sampling
    - Bounded vocab (auto-prunes least-frequent tail)
    """
    def __init__(self, max_vocab=200, base_tokens=None, temperature=0.9):
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.max_vocab = int(max_vocab)
        self.temperature = float(temperature)
        base_tokens = base_tokens or DEFAULT_BASE_TOKENS
        for t in base_tokens:
            self.observe_tokens([t])

    # ---- observation / training ----
    def observe_tokens(self, tokens, weight=1.0):
        if not tokens:
            return
        # prune new tokens if vocab too large
        for t in tokens:
            if t not in self.unigram and len(self.unigram) >= self.max_vocab:
                # drop the least common token to make room
                loser, _ = self.unigram.most_common()[:-1-1:-1][0] if self.unigram else (None, 0)
                if loser:
                    del self.unigram[loser]
                    if loser in self.bigram:
                        del self.bigram[loser]
                    for k in list(self.bigram.keys()):
                        if loser in self.bigram[k]:
                            del self.bigram[k][loser]
        # accumulate
        prev = None
        for t in tokens:
            self.unigram[t] += weight
            if prev is not None:
                self.bigram[prev][t] += weight
            prev = t

    def observe_text(self, text, weight=1.0):
        self.observe_tokens(_tokenize(text), weight=weight)

    def observe_json_dict(self, d, weight=0.5, max_items=2000):
        # take a sample to avoid loading gigantic dictionaries into the model
        items = list(d.items())
        random.shuffle(items)
        for k, v in items[:max_items]:
            self.observe_text(str(k), weight)
            self.observe_text(str(v), weight*0.5)

    # ---- decay to avoid dominance collapse ----
    def decay(self, rate=0.995):
        for k in list(self.unigram.keys()):
            self.unigram[k] *= rate
            if self.unigram[k] < 0.05:
                del self.unigram[k]
                if k in self.bigram:
                    del self.bigram[k]
                for p in list(self.bigram.keys()):
                    if k in self.bigram[p]:
                        del self.bigram[p][k]
        for p in list(self.bigram.keys()):
            row = self.bigram[p]
            for k in list(row.keys()):
                row[k] *= rate
                if row[k] < 0.05:
                    del row[k]
            if not row:
                del self.bigram[p]

    # ---- generation ----
    def _sample_from_counter(self, ctr):
        if not ctr:
            return None
        items = list(ctr.items())
        # temperature softmax
        logits = [math.log(max(c, 1e-8)) / max(self.temperature, 1e-6) for _, c in items]
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        s = sum(exps) or 1.0
        probs = [e/s for e in exps]
        r = random.random()
        acc = 0.0
        for (tok, _), p in zip(items, probs):
            acc += p
            if r <= acc:
                return tok
        return items[-1][0]

    def next_token(self, prev=None):
        if prev is None or prev not in self.bigram or not self.bigram[prev]:
            return self._sample_from_counter(self.unigram)
        return self._sample_from_counter(self.bigram[prev])

    def generate(self, min_len=1, max_len=6, repetition_penalty=1.15, recent=None):
        """
        Generate 1..N tokens with a light repetition penalty.
        `recent` is an optional set/list of tokens we want to downweight.
        """
        n = random.randint(min_len, max_len)
        out = []
        prev = None
        recent = set(recent or [])
        for _ in range(n):
            # choose candidate distribution
            row = self.unigram if (prev not in self.bigram or not self.bigram[prev]) else self.bigram[prev]
            if not row:
                break

            # temperature softmax (copy of _sample_from_counter but inline so we can penalize)
            items = list(row.items())
            logits = []
            for tok, c in items:
                base = math.log(max(c, 1e-8)) / max(self.temperature, 1e-6)
                if tok in recent or (out and tok == out[-1]):
                    base -= math.log(repetition_penalty)  # downweight repeats
                logits.append(base)

            m = max(logits)
            exps = [math.exp(x - m) for x in logits]
            s = sum(exps) or 1.0
            probs = [e/s for e in exps]
            r = random.random()
            acc = 0.0
            choice = items[-1][0]
            for (tok, _), p in zip(items, probs):
                acc += p
                if r <= acc:
                    choice = tok
                    break

            out.append(choice)
            prev = choice
        return " ".join(out)

    # reinforcement hooks (± reward adjusts counts used)
    def reinforce_sequence(self, text, reward, lr=0.2):
        # reward in [-1, 1]
        tokens = _tokenize(text)
        if not tokens:
            return
        w = lr * float(max(-1.0, min(1.0, reward)))
        prev = None
        for t in tokens:
            self.unigram[t] += w
            if prev is not None:
                self.bigram[prev][t] += w
            prev = t
        # keep things positive-ish
        for k in list(self.unigram.keys()):
            if self.unigram[k] <= 0.01:
                del self.unigram[k]
                if k in self.bigram:
                    del self.bigram[k]
        for p in list(self.bigram.keys()):
            row = self.bigram[p]
            for k in list(row.keys()):
                if row[k] <= 0.01:
                    del row[k]
            if not row:
                del self.bigram[p]