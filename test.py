import numpy as np
import matplotlib.pyplot as plt
import cma
from es import SimpleGA, CMAES, PEPG, OpenES, Pyswarms

def rastrigin(x):
  """Rastrigin test objective function"""
  x = np.copy(x)
  #x -= 10.0
  if not np.isscalar(x[0]):
    N = len(x[0])
    return -np.array([10 * N + sum(xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])
  N = len(x)
  return -(10 * N + sum(x**2 - 10 * np.cos(2 * np.pi * x)))

fit_func = rastrigin

NPARAMS = 100        # make this a 100-dimensinal problem.
NPOPULATION = 100    # use population size of 101.
MAX_ITERATION = 5000

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

pso = Pyswarms(num_params = NPARAMS,
          popsize = NPOPULATION,
          sigma_init = 1,
          weight_decay = 0.00,
          communication_topology = 'random')
pso_hist = test_solver(pso)
