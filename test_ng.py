import numpy as np
import matplotlib.pyplot as plt
import cma
from es import SimpleGA, CMAES, PEPG, OpenES, Nevergrad

def rastrigin(x):
  """Rastrigin test objective function, shifted by 10. units away from origin"""
  x = np.copy(x)
  x -= 10.0
  if not np.isscalar(x[0]):
    N = len(x[0])
    return -np.array([10 * N + sum(xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])
  N = len(x)
  return -(10 * N + sum(x**2 - 10 * np.cos(2 * np.pi * x)))

def sphere(x):
    x = np.copy(x)
    x -= 10.0
    j = (x ** 2.0).sum()

    return -j

def deceptivemultimodal(x):
    """Infinitely many local optima, as we get closer to the optimum."""
    x = np.copy(x)
    # x -= 10.0
    distance = np.sqrt(x[0] ** 2 + x[1] ** 2)
    if distance == 0.0:
        return 0.0
    angle = np.arctan(x[0] / x[1]) if x[1] != 0.0 else np.pi / 2.0
    invdistance = int(1.0 / distance) if distance > 0.0 else 0.0
    if np.abs(np.cos(invdistance) - angle) > 0.1:
        return -1.0
    return -float(distance)

fit_func = deceptivemultimodal

NPARAMS = 100        # make this a 100-dimensinal problem.
NPOPULATION = 101    # use population size of 101.
MAX_ITERATION = 5000 # run each solver for 5000 generations.

# defines a function to use solver to solve fit_func
def test_solver(solver):
  history = []
  for j in range(MAX_ITERATION):
    solutions = solver.ask()
    fitness_list = np.zeros(solver.popsize)
    for i in range(solver.popsize):
      fitness_list[i] = fit_func(solutions[i])
    solver.tell(fitness_list)
    result = solver.result() # first element is the best solution, second element is the best fitness
    history.append(result[1])
    if (j+1) % 100 == 0:
      print("fitness at iteration", (j+1), result[1])
  print("local optimum discovered by solver:\n", result[0])
  print("fitness score at this local optimum:", result[1])
  return history


x = np.zeros(NPARAMS) # 100-dimensional problem
print("This is F(0):")
print(fit_func(x))

x = np.ones(NPARAMS)*10. # 100-dimensional problem
print(fit_func(x))
print("global optimum point:\n", x)

pso = Nevergrad(optimizer = 'PSO',
                num_params = NPARAMS,
                popsize = NPOPULATION,
                sigma_init = 0.5,
                weight_decay = 0.00)
pso_hist = test_solver(pso)

cma_ng = Nevergrad(optimizer = 'CMA',
                num_params = NPARAMS,
                popsize = NPOPULATION,
                sigma_init = 0.5,
                weight_decay = 0.00)

cma_ng_hist = test_solver(cma_ng)
