from pysat.solvers import Solver
import ast
from matplotlib import pyplot as plt
from tqdm import tqdm


def read_data(filename):
    with open(filename, "r") as fid:
        formulas = [ast.literal_eval(l) for l in fid.readlines()]
    return formulas


formulas = read_data("data/formulas.txt")
time = []

# for formula in tqdm(formulas):
#     solver = Solver(name="g3", use_timer=True)
#     solver.append_formula(formula)
#     solver.solve()
#     time.append(solver.time())
#     solver.delete()

plt.plot(time)
plt.show()