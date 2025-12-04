# file: evolution/dsl.py
import random

OPS = ["READ", "WRITE", "LIST", "DRAW", "MSG", "NOOP"]

READ_PATHS = [
    "/world/notes/welcome.txt",
    "/world/notes/guide.txt",
    "/world/notes.txt",
    "/world/dictionary.json"
]

WRITE_TARGETS = [
    "/world/notes.txt",     # shared world notes
    "/log.txt",             # private home log
    "/scratch.txt"          # private home scratch
]

LIST_PATHS = [
    "/world",
    "/world/notes",
    "/"
]

WORDS = ["hi", "ok", "note", "x", "y", "yes", "no", "maybe"]


def random_instruction():
    op = random.choice(OPS)

    if op == "NOOP":
        return ("NOOP",)

    if op == "READ":
        return ("READ", random.choice(READ_PATHS))

    if op == "LIST":
        return ("LIST", random.choice(LIST_PATHS))

    if op == "WRITE":
        return ("WRITE",
                random.choice(WRITE_TARGETS),
                random.choice(WORDS))

    if op == "DRAW":
        return ("DRAW",
                random.randint(0, 63),
                random.randint(0, 63))

    if op == "MSG":
        return ("MSG",
                random.randint(0, 29),
                {"note": random.choice(["a", "b", "c", "ping", "hello"])})