import numpy as np
import matplotlib.pyplot as plt
import cma
from es import SimpleGA, CMAES, PEPG, OpenES, PSO, modified_PSO, PSO_CMA_ES, local_PSO, PSO_CMA_ES3

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
    x -= 10.0
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
    #import pdb; pdb.set_trace()
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
                        c1 = 0.001,
                        c2 = 0.003,
                        w = 0.001,
                        popsize = NPOPULATION,
                        sigma_init = 0.5,
                        weight_decay = 0.00,
                        min_pop_std = 0.3)

pso_cma_es_history = test_solver(pso_cma_es)

# pso_cma_es3 = PSO_CMA_ES3(NPARAMS,
#                         c1 = 0.001,
#                         c2 = 0.003,
#                         w = 0.001,
#                         popsize = NPOPULATION,
#                         sigma_init = 0.5,
#                         weight_decay = 0.00,
#                         min_pop_std = 10000,
#                         slack_calls = 5,
#                         pso_sigma_init = 0.1)
#
# pso_cma_es3_history = test_solver(pso_cma_es3)


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