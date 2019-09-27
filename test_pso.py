import numpy as np
import matplotlib.pyplot as plt
import cma
from es import SimpleGA, CMAES, PEPG, OpenES, PSO, modified_PSO, PSO_CMA_ES, local_PSO, PSO_CMA_ES2

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
    x -= 10.0
    j = (x ** 2.0).sum()

    return -j

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
print(fit_func(x))

x = np.ones(NPARAMS)*10. # 100-dimensional problem
print(fit_func(x))
print("global optimum point:\n", x)

pso_cma_es = PSO_CMA_ES(NPARAMS,
                        c1 = 0.5,
                        c2 = 0.5,
                        w = 0.9,
                        popsize = NPOPULATION,
                        sigma_init = 0.5,
                        weight_decay = 0.00,
                        min_pop_std = 0.1)

pso_cma_es_history = test_solver(pso_cma_es)

# pso_cma_es2 = PSO_CMA_ES2(NPARAMS,
#                         c1 = 0.5,
#                         c2 = 0.5,
#                         w = 0.9,
#                         popsize = NPOPULATION,
#                         sigma_init = 0.5,
#                         weight_decay = 0.00,
#                         min_pop_std = 0.7)
#
# pso_cma_es2_history = test_solver(pso_cma_es2)

#
# pso = PSO(NPARAMS,
#          c1 = 0.5,
#          c2 = 0.5,
#          w = 0.9,
#          popsize = NPOPULATION,
#          sigma_init = 0.5,
#          weight_decay = 0.00)
#
# pso_history = test_solver(pso)

# local_pso = local_PSO(NPARAMS,
#                 c1 = 0.3,
#                 c2 = 0.1,
#                 w = 0.9,
#                 popsize = NPOPULATION,
#                 sigma_init = 0.9,
#                 weight_decay = 0.00,
#                 neighbours = 3)
#
# local_pso_history = test_solver(local_pso)

# m_pso = modified_PSO(NPARAMS,
#          c1 = 0.5,
#          c2 = 0.5,
#          w = 0.9,
#          popsize = NPOPULATION,
#          sigma_init = 0.5,
#          weight_decay = 0.00)
#
# m_pso_history = test_solver(m_pso)
