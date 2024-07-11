import numpy as np


def normalized_p(p, add: bool, exp: bool):
  resize = (lambda x: np.exp(x) - 1) if exp else (lambda x: x)

  normalize = lambda x: x / sum(x)
  p = normalize(p)
  p = resize(p)
  p = normalize(p)
  if not add:
    p = normalize(1-p)
  return p


def get_out_degrees_weight_function(G, exp=False):
  # # I from 'infected'
  # neighbors = G.edges_from(node)
  # if not neighbors: return np.inf
  # not_infected_neighbors = neighbors.difference(I)
  # return len(not_infected_neighbors)

  def inner(A, out, add: bool):
    p = np.zeros(len(A))
    for i, a in enumerate(A):
      neighbors = G.edges_from(a)
      out_neighbors = filter(lambda x: x in out, neighbors)
      num_out_neighbors = sum(1 for _ in out_neighbors)
      p[i] = num_out_neighbors
    return normalized_p(p, add, exp)

  return inner


# def get_t_difference_weight_function(G, t, exp=False):
#   # # I from 'infected'
#   # neighbors = G.edges_from(node)
#   # if not neighbors: return 0
#   # infected_neighbors = neighbors & I
#   # frac1 = min(t, len(infected_neighbors) / len(neighbors))
#   # frac2 = min(t, len(infected_neighbors) + 1 / len(neighbors))
#   # return frac2 - frac1
#   resize = np.exp if exp else lambda x: x
#
#   def inner(A, out, add: bool):
#     p = np.zeros(len(A))
#     for i, a in enumerate(A):
#       neighbors = G.edges_from(a)
#       in_neighbors = neighbors.diffrence(out)
#       if add:
#         t_curr = len(in_neighbors) / len(neighbors)
#         t_new = (len(in_neighbors) + 1) / len(neighbors)
#       else:
#         t_curr = (len(in_neighbors) - 1) / len(neighbors)
#         t_new = len(in_neighbors) / len(neighbors)
#       p[i] = resize(t_new - t_curr)
#
#   return inner
