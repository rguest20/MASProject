import os


POP_SIZE = 30
KEEP_RATIO = 0.5
TARGET = 47.0
# Short, inspectable experiments make behavioural regressions visible. Raise
# this deliberately for longer studies after reviewing the 30-turn report.
GENERATIONS = 30000
DEFAULT_DIMS = 32
MIN_DIMS = 8
MAX_DIMS = 256
DIMS = int(os.environ.get("MAS_SEMANTIC_DIMS", DEFAULT_DIMS))
if not MIN_DIMS <= DIMS <= MAX_DIMS:
    raise ValueError(f"MAS_SEMANTIC_DIMS must be between {MIN_DIMS} and {MAX_DIMS}")
