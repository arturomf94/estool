import numpy as np
import matplotlib.pyplot as plt
import cma
from es import SimpleGA, CMAES, PEPG, OpenES, PSO, modified_PSO, PSO_CMA_ES

def rastrigin(x):
  """Rastrigin test objective function, shifted by 10. units away from origin"""
  x = np.copy(x)
  x -= 10.0
  if not np.isscalar(x[0]):
    N = len(x[0])
    return -np.array([10 * N + sum(xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])
  N = len(x)
  return -(10 * N + sum(x**2 - 10 * np.cos(2 * np.pi * x)))

fit_func = rastrigin

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
print(rastrigin(x))

x = np.ones(NPARAMS)*10. # 100-dimensional problem
print(rastrigin(x))
print("global optimum point:\n", x)

pso_cma_es = PSO_CMA_ES(NPARAMS,
                        c1 = 0.1,
                        c2 = 0.9,
                        w = 0.9,
                        popsize = NPOPULATION,
                        sigma_init = 0.5,
                        weight_decay = 0.00,
                        min_pop_std = 0.75)

pso_cma_es_history = test_solver(pso_cma_es)

# ga = SimpleGA(NPARAMS,                # number of model parameters
#                sigma_init=0.5,        # initial standard deviation
#                popsize=NPOPULATION,   # population size
#                elite_ratio=0.1,       # percentage of the elites
#                forget_best=False,     # forget the historical best elites
#                weight_decay=0.00,     # weight decay coefficient
#               )
#
# ga_history = test_solver(ga)
