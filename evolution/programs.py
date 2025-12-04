import random
import math

# ============================================================
# PROGRAM MODEL (Phase-3 safe)
# ============================================================

# Only these ops are supported by run_program().
VALID_OPS = ["w", "m", "s", "d", "x"]

# Value range for constants.
CONST_RANGE = (-10.0, 10.0)

# Hard max program length
MAX_PROGRAM_LEN = 12


# ------------------------------------------------------------
# Helper: safe numeric clamp
# ------------------------------------------------------------
def safe(v, lo=-1e6, hi=1e6):
    if v != v:              # NaN
        return 0.0
    if v == float('inf'):   # +∞
        return hi
    if v == float('-inf'):  # -∞
        return lo
    return max(lo, min(hi, v))


# ------------------------------------------------------------
# Generate a random instruction guaranteed valid
# ------------------------------------------------------------
def random_instruction():
    op = random.choice(VALID_OPS)
    v = round(random.uniform(*CONST_RANGE), 3)
    return (op, v)


# ------------------------------------------------------------
# Generate a random program
# ------------------------------------------------------------
def make_random_program(length=5):
    length = max(1, min(int(length), MAX_PROGRAM_LEN))
    return [random_instruction() for _ in range(length)]


# ------------------------------------------------------------
# SANITISER — ensures program is a list of 2-tuples (op, val)
# ------------------------------------------------------------
def sanitize_program(prog):
    clean = []
    for item in prog:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            op, v = item
            if isinstance(op, str) and isinstance(v, (int, float)) and op in VALID_OPS:
                clean.append((op, float(v)))
                continue

        # If anything is malformed → replace with safe instruction
        clean.append(random_instruction())

    # clamp length
    clean = clean[:MAX_PROGRAM_LEN]
    if len(clean) == 0:
        clean = [random_instruction()]
    return clean


# ------------------------------------------------------------
# PROGRAM EXECUTION ENGINE
# ------------------------------------------------------------
def run_program(prog):
    acc = 0.0

    for item in prog:
        # Validation layer
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue

        op, v = item
        if op not in VALID_OPS:
            continue
        if not isinstance(v, (int, float)):
            continue

        # Safe execution
        if op == "w":      # add
            acc += v
        elif op == "m":    # subtract
            acc -= v
        elif op == "s":    # square-add scaled
            acc += (v * v) * 0.001
        elif op == "d":    # damp
            acc *= 0.5
        elif op == "x":    # invert
            acc = -acc

        acc = safe(acc)

    return acc


# ------------------------------------------------------------
# FITNESS FUNCTION (this is legacy Phase-1 compatible)
# ------------------------------------------------------------
def fitness(program, target=47.0):
    result = run_program(program)
    result = math.log1p(abs(result))  # keeps extreme outputs tame
    closeness = -abs(result - target)
    penalty = -0.01 * len(program)
    return closeness + penalty


# ------------------------------------------------------------
# MUTATION
# ------------------------------------------------------------
def mutate_program(prog, p_mut=0.2, p_add=0.1, p_del=0.1):
    prog = sanitize_program(prog)
    new = []

    for instr in prog:
        if random.random() < p_mut:
            new.append(random_instruction())
        else:
            # Keep instruction safe
            op, v = instr
            new.append((op, float(v)))

    # Possibly add
    if random.random() < p_add and len(new) < MAX_PROGRAM_LEN:
        new.append(random_instruction())

    # Possibly delete
    if random.random() < p_del and len(new) > 1:
        idx = random.randrange(len(new))
        new.pop(idx)

    return sanitize_program(new)


# ------------------------------------------------------------
# STABILITY SCORE
# ------------------------------------------------------------
def stability_score(program):
    program = sanitize_program(program)

    results = []
    for _ in range(5):
        perturbed = []
        for op, val in program:
            new_val = val + random.gauss(0, 0.01)
            perturbed.append((op, new_val))
        results.append(run_program(perturbed))

    cleaned = [r for r in results if isinstance(r, (int, float))]
    if len(cleaned) < 2:
        return -100.0

    mean = sum(cleaned) / len(cleaned)
    var = sum((x - mean) ** 2 for x in cleaned) / len(cleaned)
    return -var  # closer to 0 is more stable


# ------------------------------------------------------------
# NOVELTY SCORE
# ------------------------------------------------------------
def novelty_score(program, population):
    if not population:
        return 0.0

    program = sanitize_program(program)
    population = [sanitize_program(p) for p in population]

    # centroid
    min_len = min(len(program), *(len(p) for p in population))
    centroid = []
    for i in range(min_len):
        ops = [p[i][0] for p in population]
        vals = [p[i][1] for p in population]

        common_op = max(set(ops), key=ops.count)
        mean_val = sum(vals) / len(vals)
        centroid.append((common_op, mean_val))

    # novelty difference
    score = 0.0
    for (op_a, v_a), (op_b, v_b) in zip(program, centroid):
        if op_a != op_b:
            score += 1.0
        score += abs(v_a - v_b) * 0.02

    return score
