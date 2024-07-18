# Skeleton file for HW2 question 9
# =====================================
# IMPORTANT: You are NOT allowed to modify the method signatures
# (i.e. the arguments and return types each function takes).
# We will pass your grade through an autograder which expects a specific format.
# =====================================


# Do not include any other files or an external package, unless it is one of
# [numpy, pandas, scipy, matplotlib, random]
# please contact us before sumission if you want another package approved.
import time, math, numpy as np
import matplotlib.pyplot as plt
from queue import Queue
import random
from dataclasses import dataclass, field
import multiprocessing as mp


# Implement the methods in this class as appropriate. Feel free to add other methods
# and attributes as needed. You may/should reuse code from previous HWs when applicable.
class UndirectedGraph:
    def __init__(self, number_of_nodes):
        """Assume that nodes are represented by indices/integers between 0 and number_of_nodes - 1."""
        self.n = number_of_nodes
        self.edges = {node: set() for node in range(number_of_nodes)}
        self.distances = {}

    def add_edge(self, nodeA, nodeB):
        """Adds an undirected edge to the graph, between nodeA and nodeB. Order of arguments should not matter"""
        if nodeA == nodeB:
            return
        self.edges[nodeA].add(nodeB)
        self.edges[nodeB].add(nodeA)
        self.distances.clear()

    def edges_from(self, nodeA):
        """This method shold return a list of all the nodes nodeB such that nodeA and nodeB are
        connected by an edge"""
        # TODO: Need to verify that copying the set each time does not come with a runtime panalty, since it is used in a loop! Edit: no diffrence
        # return self.edges[nodeA] # FIXME: should we do this instead # time=39.14
        # return list(self.edges[nodeA]) # time=39.09
        return self.edges[nodeA]  # changed

    def check_edge(self, nodeA, nodeB):
        """This method should return true is there is an edge between nodeA and nodeB, and false otherwise"""
        return nodeB in self.edges[nodeA]

    def number_of_nodes(self):
        """This method should return the number of nodes in the graph"""
        return self.n

    def single_source_bfs(self, sNode):
        visited = set()
        distances = [-1] * self.n
        parents = distances.copy()

        visited.add(sNode)
        distances[sNode] = 0

        nodes_to_visit = Queue()

        nodes_to_visit.put(sNode)

        while nodes_to_visit.qsize():
            popped = nodes_to_visit.get()
            for neighbor in self.edges[popped]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[popped] + 1
                    parents[neighbor] = popped
                    nodes_to_visit.put(neighbor)

        self.distances[sNode] = distances

    def get_distance(self, node1, node2):
        if node1 not in self.distances and node2 not in self.distances:
            self.single_source_bfs(min(node1, node2))

        if node1 in self.distances:
            return self.distances[node1][node2]
        if node2 in self.distances:
            return self.distances[node2][node1]

    def is_connected(self):
        if 0 not in self.distances:
            self.single_source_bfs(0)
        return self.distances[0].count(-1) == 0


def create_fb_graph(filename="facebook_combined.txt"):
    """This method should return a undirected version of the facebook graph as an instance of the UndirectedGraph class.
    You may assume that the input graph has 4039 nodes."""
    num_nodes = 4039
    graph = UndirectedGraph(num_nodes)
    for line in open(filename):
        i, j = map(int, line.strip().split(" "))
        # i, j = list(map(int, line.strip().split(" ")))
        graph.add_edge(i, j)
    return graph


# === Problem 9(a) ===


def contagion_brd(G, S, t):
    """Given an UndirectedGraph G, a list of adopters S (a list of integers in [0, G.number_of_nodes - 1]),
    and a float threshold t, perform BRD as follows:
    - Permanently infect the nodes in S with X
    - Infect the rest of the nodes with Y
    - Run BRD on the set of nodes not in S
    Return a list of all nodes infected with X after BRD converges."""
    assert 0 <= t <= 1, "t must be in [0, 1]"
    assert all(0 <= i < G.n for i in S), "S must be a list of integers in [0, G.number_of_nodes - 1]"

    adopters_set = set(S)
    is_X = [(i in adopters_set) for i in range(G.n)]
    # is_X = list(map(lambda idx: idx in adopters_set, range(G.n)))
    needs_addressing = set(range(G.n)).difference(adopters_set)

    def brd_step(i):
        neighbors = G.edges_from(i)

        if not neighbors:
            return

        frac = sum(is_X[n] for n in neighbors) / len(neighbors)
        if frac > t and not is_X[i]:
            is_X[i] = True
            needs_addressing.update(
                {n for n in neighbors if not is_X[n] and n not in adopters_set}
            )
        if frac < t and is_X[i]:
            is_X[i] = False
            needs_addressing.update(
                {n for n in neighbors if is_X[n] and n not in adopters_set}
            )
        # X_neighbors = set(filter(lambda neighbor: is_X[neighbor], neighbors))
        # frac = len(X_neighbors) / len(neighbors)
        # if frac > t and not is_X[i]:
        #     is_X[i] = True
        #     needs_addressing.update(
        #         set(neighbors).difference(X_neighbors).difference(adopters_set)
        #     )
        #
        # if frac < t and is_X[i]:
        #     is_X[i] = False
        #     needs_addressing.update(set(X_neighbors).difference(adopters_set))

    while needs_addressing:
        brd_step(needs_addressing.pop())

    return [i for i in range(G.n) if is_X[i]]
    # return list(filter(lambda i: is_X[i], range(G.n)))


fig4_1_left = UndirectedGraph(4)
fig4_1_left.add_edge(0, 1)
fig4_1_left.add_edge(1, 2)
fig4_1_left.add_edge(2, 3)


fig4_1_right = UndirectedGraph(7)
fig4_1_right.add_edge(0, 1)
fig4_1_right.add_edge(1, 2)
fig4_1_right.add_edge(1, 3)
fig4_1_right.add_edge(3, 4)
fig4_1_right.add_edge(3, 5)
fig4_1_right.add_edge(5, 6)


def q_completecascade_graph_fig4_1_left():
    """Return a float t s.t. the left graph in Figure 4.1 cascades completely."""
    S = [0, 1]
    t = 0.49
    infected = contagion_brd(fig4_1_left, S, t)
    fully_infected = len(infected) == fig4_1_left.n
    if fully_infected:
        print(f"Success: graph fig4_1_left was fully infected with {t=}")
    else:
        print(f"Failure: graph fig4_1_left was not fully infected with {t=}")
    return t


def q_incompletecascade_graph_fig4_1_left():
    """Return a float t s.t. the left graph in Figure 4.1 does not cascade completely."""
    S = [0, 1]
    t = 0.51
    infected = contagion_brd(fig4_1_left, S, t)
    fully_infected = len(infected) == fig4_1_left.n
    if fully_infected:
        print(f"Failure: graph fig4_1_left was fully infected with {t=}")
    else:
        print(f"Success: graph fig4_1_left was not fully infected with {t=}")
    return t


def q_completecascade_graph_fig4_1_right():
    """Return a float t s.t. the right graph in Figure 4.1 cascades completely."""
    S = [0, 1, 2]
    t = 0.32
    infected = contagion_brd(fig4_1_right, S, t)
    fully_infected = len(infected) == fig4_1_right.n
    if fully_infected:
        print(f"Success: graph fig4_1_right was fully infected with {t=}")
    else:
        print(f"Failure: graph fig4_1_right was not fully infected with {t=}")
    return t


def q_incompletecascade_graph_fig4_1_right():
    """Return a float t s.t. the right graph in Figure 4.1 does not cascade completely."""
    S = [0, 1, 2]
    t = 0.34
    infected = contagion_brd(fig4_1_right, S, t)
    fully_infected = len(infected) == fig4_1_right.n
    if fully_infected:
        print(f"Failure: graph fig4_1_right was fully infected with {t=}")
    else:
        print(f"Success: graph fig4_1_right was not fully infected with {t=}")
    return t


def sanity_checks():
    q_completecascade_graph_fig4_1_left()
    q_incompletecascade_graph_fig4_1_left()
    q_completecascade_graph_fig4_1_right()
    q_incompletecascade_graph_fig4_1_right()


##### BONUS 2 - SIMULATED ANNEALING ######
def run_optimizer_par(G, t, duration=80,
                      exp_or_None_weight_function=True,
                      num_processes=1):
  """
  Run the optimizer in parallel
  G - the graph
  t - the t-fraction
  exp_or_None_weight_function
    None - runs without any weight function
    True - runs with the exponential weight function (large weight are exponantialy more likely)
    False - runs with the linear weight function (large weight are linearly more likely)
  """
  if num_processes == 1:
    return run_optimizer(G, t, duration, exp_or_None_weight_function)

  with mp.Pool(processes=num_processes) as pool:
    s_es = pool.starmap(
      run_optimizer,
      [(G, t, duration, exp_or_None_weight_function) for _ in
       range(num_processes)]
    )

  return min(s_es, key=lambda s: loss_func(State(G, t, set(), S=s)))


def run_optimizer(G, t, duration, exp_or_None_weight_function=None):
  """
  Run the optimizer sequentially
  G - the graph
  t - 100 * t is the batch size
  exp_or_None_weight_function
    None - runs without any weight function
    True - runs with the exponential weight function (large weight are exponantialy more likely)
    False - runs with the linear weight function (large weight are linearly more likely)
  """
  shared_prev_states = set()
  run = lambda tmp, dur, st: _simulated_annealing(
    Temp=tmp,
    initial_state=st,
    alpha=0.99,
    duration=dur,
    loss=loss_func,
    accept=_accept_func,
    update=_update_func,
  )

  butch_size = max(1, 100 * t)  # number of nodes to add / remove each time
  # third = lambda x: (duration * 2) // 3
  # duration = third(1 + duration / 2)
  duration = 1 + duration // 2

  state = State(G, t, shared_prev_states,
                exp_or_None_weight_function,
                butch_size=butch_size)

  state, temp = run(100, duration, state)

  while duration >= 1:
    duration, batch_size = duration // 2, max(1, butch_size // 3)
    state, temp = run(temp, duration,
                      State(G, t, shared_prev_states,
                            exp_or_None_weight_function,
                            S=state.S,
                            butch_size=butch_size),
                      )
  return state.S


def loss_func(state):
  """
  adding one infected is better always better
  given the same amount of infected, the smaller S is, the better
  """
  G, S, I_comp = state.G, state.S, state.I_comp
  number_of_non_infected = len(I_comp)  # in N
  fraction_of_S = max(0, len(S) - 1) / G.n  # in [0, 1)
  return number_of_non_infected + fraction_of_S


@dataclass(frozen=True, repr=False)
class MySet(set):
  _hash: int = field(init=False)
  _list: list = field(init=False)

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    object.__setattr__(self, '_hash', hash(frozenset(self)))
    object.__setattr__(self, '_list', list(self))

  def __hash__(self):
    return self._hash

  def as_list(self):
    return self._list

  def __repr__(self):
    return '{' + ', '.join(map(str, self)) + '}'


@dataclass
class Data:
  a: int
  b: int
  butch_size: int

  def calc(self, s_size):
    if not self.bootstrap(): return np.random.randint(1, max(self.butch_size, 2))
    middle = (self.a + self.b) // 2
    return max(1, np.abs(s_size - middle))

  def bootstrap(self):
    return self.b != self.a


class State:
  def __init__(self, G, t,
               shared_prev_states,
               exp_or_None=None, S=None,
               _data=None, butch_size=1):
    self.G, self.t = G, t
    self.exp_or_None = exp_or_None
    self.shared_prev_states = shared_prev_states

    self.S = MySet(S) if S is not None else MySet()
    shared_prev_states.add(self.S)

    self.I = contagion_brd(self.G, self.S, self.t)
    self.I_comp = [i for i in range(self.G.n) if i not in self.I]
    self.data = self._calc_data(_data, butch_size)

  def is_cascading(self):
    return len(self.I_comp) == 0

  def _calc_data(self, data, butch_size):
    if data is None:
      return Data(0, self.G.n, butch_size)

    if not data.bootstrap():
      return data

    if self.is_cascading():
      return Data(data.a, len(self.S), data.butch_size)
    else:
      return Data(len(self.S), data.b, data.butch_size)

  def new_state(self, S=None, _new_game=True):
    return State(self.G, self.t, self.shared_prev_states, self.exp_or_None,
                 S, _data=None if _new_game else self.data)

  @property
  def S_weights(self):
    if self.exp_or_None is None: return None
    comp = self.I_comp if self.I_comp else self.S_comp
    weight_func = _out_degrees_weight_function(self.G, exp=self.exp_or_None)
    return weight_func(self.S, comp, add=False)

  @property
  def C_weights(self):
    if self.exp_or_None is None: return None
    weight_func = _out_degrees_weight_function(self.G, exp=self.exp_or_None)
    return weight_func(self.I_comp, self.I_comp, add=True)

  @property
  def S_comp(self):
    return [i for i in range(self.G.n) if i not in self.S]

  def __repr__(self):
    return f'Cascading={self.is_cascading()}_sLen={len(self.S)}'


def _accept_func(state, delta_l, Temp) -> object:
  if state.data.bootstrap():
    return True

  val = - delta_l / Temp
  return np.random.random() < math.exp(val)


def _random_add_node(state) -> State:
  size = max(1, min(state.data.calc(len(state.S)),
                    len(state.I_comp) // 2,
                    max(8, int(np.ceil(len(state.S) / 2)))))
  new_s = state.S
  for _ in range(100):
    if sum(x != 0 for x in state.C_weights) < size:
      nodes_to_add = np.array(state.I_comp)[np.array(state.C_weights) != 0]
    else:
      nodes_to_add = np.random.choice(state.I_comp, size, False,
                                    p=state.C_weights)
    new_s = frozenset(set(nodes_to_add) | state.S)
    if hash(new_s) not in state.shared_prev_states:
      break

  return state.new_state(new_s, _new_game=False)


def _random_remove_node(state) -> State:
  size = max(1, min(state.data.calc(len(state.S)), len(state.S) // 2))
  new_s = state.S
  for _ in range(100):
    node_to_remove = np.random.choice(state.S.as_list(), size, False,
                                      p=state.S_weights)
    new_s = frozenset(state.S - set(node_to_remove))
    if hash(new_s) not in state.shared_prev_states:
      break

  return state.new_state(new_s, _new_game=False)


def _update_func(state):
  return _random_remove_node(state) \
    if state.is_cascading() \
    else _random_add_node(state)


def _normalized_p(p, add: bool, exp):
  resize = (lambda x: np.exp(x) - 1) if exp else (lambda x: x)

  normalize = lambda x: x / sum(x) if sum(x) != 0 else x
  p = normalize(p)
  p = resize(p)
  p = normalize(p)
  if not add:
    p = normalize(1 - p)
  if np.all(p == 0):
    p = np.array([1/len(p) for _ in p])
  return p


def _out_degrees_weight_function(G, exp=False):
  # # I from 'infected'
  # neighbors = G.edges_from(node)
  # if not neighbors: return np.inf
  # not_infected_neighbors = neighbors.difference(I)
  # return len(not_infected_neighbors)

  def inner(A, out, add):
    out_as_set = set(out)
    p = np.zeros(len(A))
    for i, a in enumerate(A):
      neighbors = G.edges_from(a)
      out_neighbors = filter(lambda x: x in out_as_set, neighbors)
      num_out_neighbors = sum(1 for _ in out_neighbors)
      p[i] = num_out_neighbors
    return _normalized_p(p, add, exp)

  return inner


def _simulated_annealing(Temp, initial_state, alpha,
                         duration, loss, accept,
                         update):
  assert 0 < alpha < 1

  s_best = s_curr = initial_state
  l_best = l_curr = loss(s_curr)

  time_to_end = time.time() + duration

  while l_best > 0 and time.time() < time_to_end:
    Temp = max(0.001, Temp * alpha)

    s_new = update(s_curr)
    l_new = loss(s_new)

    if l_new >= l_curr and not accept(s_curr, l_new - l_curr, Temp):
      continue

    s_curr, l_curr = s_new, l_new

    if l_curr < l_best:
      s_best, l_best = s_curr, l_curr

  return s_best, Temp


def _test():
  from hw2_p9 import create_fb_graph
  G = create_fb_graph()
  t = 0.4
  s = run_optimizer(G, t,
                    duration=10,
                    exp_or_None_weight_function=True)
  # None(80): 844
  # With(80): 814
  i = contagion_brd(G, s, t)
  print(len(i) == G.n, len(s), s)


def _test_par():
  from hw2_p9 import create_fb_graph
  G = create_fb_graph()
  t = 0.4
  s = run_optimizer_par(G, t,
                        duration=10,
                        exp_or_None_weight_function=True,
                        num_processes=8)
  # with(60): 866
  # with(80): 810, 852
  i = contagion_brd(G, s, t)
  print(len(i) == G.n, len(s), s)
############################################

# === OPTIONAL: Bonus Question 2 === #
def min_early_adopters(G, q):
    """Given an undirected graph G, and float threshold t, approximate the
    smallest number of early adopters that will call a complete cascade.
    Return an integer between [0, G.number_of_nodes()]"""
    return len(run_optimizer_par(G, q, duration=10))
############################################

def main():
    sanity_checks()
    fb_graph = create_fb_graph()
    # === Problem 9(b) === #
    print("\nQ9b\n")
    threshold = 0.1
    k = 10
    T = 100
    num_infected_total = 0
    times_infected = 0
    measurements = []
    for i in range(T):
        S = random.sample(range(fb_graph.n), k)
        infected = contagion_brd(fb_graph, S, threshold)
        num_infected = len(infected)
        num_infected_total += num_infected
        if num_infected == fb_graph.n:
            times_infected += 1
            # print(f"graph was fully infected on iteration {i}")
        measurements.append(num_infected)
    avg_num_infected = num_infected_total / T
    print(f"{avg_num_infected=}")
    print(f"infected {times_infected}/{T} times.")

    plt.hist(measurements, bins=50)
    plt.xlabel("Number of infected nodes")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of number of infected nodes for k={k} and t={threshold}")
    plt.show()

    # === Problem 9(c) === #

    print("\nQ9c\n")
    ts = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    ks = list(range(0, 251, 10))
    T = 10
    for threshold in ts:
        infected_by_k = []
        for k in ks:
            num_infected_total = 0
            times_infected = 0
            for i in range(T):
                S = random.sample(range(fb_graph.n), k)
                infected = contagion_brd(fb_graph, S, threshold)
                num_infected = len(infected)
                num_infected_total += num_infected
                times_infected += num_infected == fb_graph.n
            avg_num_infected = num_infected_total / T
            infected_by_k.append(avg_num_infected)
            print(
                f"\r{threshold=}, {k=}, {avg_num_infected=}, fully infected {times_infected}/{T} times.",
                end="",
            )
        print()
        plt.plot(ks, infected_by_k, label=f"t={threshold}")
    plt.legend()
    plt.show()


    # === OPTIONAL: Bonus Question 2 === # FIXME: uncomment to run bonus 2
    min_s_size = []
    thresholds = np.linspace(0.1, 0.9, 10)
    for t in thresholds:
        print(f'\r running optimizer for t = {t:.2f}..', end='')
        len_s = min_early_adopters(fb_graph, t)
        # len(run_optimizer_par(fb_graph, t, duration=10))
        print(f'\r running optimizer for t = {t:.2f}.. size of S = {len_s}')

        min_s_size.append(len_s)

    plt.plot(thresholds, min_s_size)
    plt.xlabel('Threshold (t)')
    plt.ylabel('Minimal size of S')
    plt.title('Approximating minimal S size per threshold')
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    main()
