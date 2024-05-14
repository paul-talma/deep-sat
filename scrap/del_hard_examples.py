from pysat.examples.genhard import CB
from pysat.examples.genhard import PHP
from pysat.solvers import Solver
from matplotlib import pyplot as plt

# mutilated chessboard
PROBLEM_SIZE = 7
mcb = CB(PROBLEM_SIZE, exhaustive=True, verb=False)
mcb.to_file("data/mcb.txt")

# solve mcb
solve_mcb = False
if solve_mcb:
    glucose3 = Solver(name="g3", use_timer=True)
    glucose3.append_formula(mcb)
    print(f"Chessboard formula is sat: {glucose3.solve()}")
    print(f"Time taken: {glucose3.time()}" + "\n")
    glucose3.delete()

# pigeonhole principle
n_holes = 6
hole_capacity = 2
php = PHP(n_holes, hole_capacity, verb=False)
php.to_file("data/php.txt")

# solve PHP
time = []
for n in range(5, 9):
    php = PHP(n, hole_capacity, verb=False)
    solve_php = True
    if solve_php:
        glucose3 = Solver(name="g3", use_timer=True)
        glucose3.append_formula(php)
        glucose3.solve()
        time.append(glucose3.time())
        glucose3.delete()

plt.plot(time)
plt.show()
