import time, random, math
from hw2_p9 import contagion_brd


def run_optimizer(G, t, duration, add_func=None, remove_func=None):
  shared_prev_states = set()
  update_function = _get_update_func(
    add_func=add_func or _random_add,
    remove_func=remove_func or _random_remove,
  )
  run = lambda s: _simulated_annealing(
    Temp=100,
    state=s,
    alpha=0.99,
    duration=duration,
    loss=loss_func,
    accept=_accept_func,
    update=update_function,
  )

  time_to_end = time.time() + duration
  best_state = run(State(G, t, shared_prev_states, set()))
  while loss_func(best_state) > 0 and time.time() < time_to_end:
    new_state = run(best_state.new_state(set()))

    if loss_func(new_state) < loss_func(best_state):
      best_state = new_state

  return best_state

def loss_func(state):
  """
  adding one infected is better always better
  given the same amount of infected, the smaller S is, the better
  """
  G, S, I = state.G, state.S, state.I
  number_of_non_infected = G.n - len(I)  # in N
  fraction_of_S = len(S) / (G.n + 1)  # in [0, 1)
  return number_of_non_infected + fraction_of_S

class State:
  def __init__(self, G, t, shared_prev_states, S):
    self.G = G
    self.t = t
    self.shared_prev_states = shared_prev_states
    self.S = S
    self.I = contagion_brd(G, S, t)

    shared_prev_states.add(S)

  def is_cascading(self):
    return len(self.S) + len(self.I) == self.G.n

  def new_state(self, S):
    return State(self.G, self.t, self.shared_prev_states, S)





def _accept_func(delta_l: float, Temp: float):
  val = - delta_l / Temp
  return random.random() < math.exp(val)


def _random_remove(state) -> State:
  # TODO: completely naive implementation
  node_to_remove = None
  while not node_to_remove:
    node_to_remove = random.choice(state.S)
    if node_to_remove in state.shared_prev_states:
      node_to_remove = None
  return state.new_state(state.S - {node_to_remove})


def _random_add(state) -> State:
  node_to_add = None
  while not node_to_add:
    node_to_add = random.choice(state.I)
    if node_to_add in state.shared_prev_states:
      node_to_add = None
  return state.new_state(state.S | {node_to_add})

def _get_update_func(add_func: callable, remove_func: callable):
  def inner(state):
    return add_func(state) if state.is_cascading() else remove_func(state)
  return inner

def _simulated_annealing(Temp: float, state: State, alpha: float,
                         duration: int, loss: callable, accept: callable,
                         update: callable) -> State:
  assert 0 < alpha < 1

  s_best = s_curr = update(state, Temp)
  l_best = l_curr = loss(state)

  time_to_end = time.time() + duration

  while l_curr > 0 and Temp <= 0.0001 and time.time() < time_to_end:
    s_new = update(s_curr, Temp)
    l_new = loss(s_new)
    delta_loss = l_new - l_curr
    if delta_loss >= 0 and not accept(delta_loss, Temp):
      continue
    s_curr, l_curr = s_new, l_new
    if l_curr < l_best:
      s_best, l_best = s_curr, l_curr
    Temp *= alpha
  return s_best
