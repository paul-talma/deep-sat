import ast

import numpy as np
import pysat
from pysat.solvers import Glucose42

gluc = Glucose42()

formulas = []
with open("formulas.txt", "r") as fid:
    with open("formulas_target.txt", "w") as tar:
        for line in fid.readlines():
            form = ast.literal_eval(line)
            with Glucose42(bootstrap_with=form) as g:
                sat = 1 * g.solve()
                sat = str(sat)
                tar.write(sat + "\n")

