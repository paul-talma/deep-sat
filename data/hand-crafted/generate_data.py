# TODO: filter contradictory clauses
# TODO: filter clauses with redundant variables
# TODO: add nice features: backbones, forced generation, etc.

import random
from random import sample

# random.seed(0)
MIN_CLAUSES = 100
MAX_CLAUSES = 200
VARIABLES = range(1, 30)


def generate_formula(variables, n_clauses):
    formula = []
    for _ in range(n_clauses):
        formula.append(generate_clause(variables))
    return formula


def generate_clause(variables):
    clause = sample(variables, 3)
    negate = random.randint(0, 3)
    indices = sample(range(3), negate)
    for i in indices:
        clause[i] = -clause[i]
    return clause


with open("data/hand-crafted/formulas.txt", "w") as fid:
    for i in range(MIN_CLAUSES, MAX_CLAUSES):
        for _ in range(10):
            form = generate_formula(VARIABLES, n_clauses=i)
            fid.write("[" + ", ".join(map(str, form)) + "]" + "\n")
