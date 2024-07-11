import time, math, numpy as np
from dataclasses import dataclass, field

from hw2_p9 import contagion_brd


def run_optimizer(G, t, duration, weight_function=None):
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

  butch_size = 100 * t  # number of nodes to add / remove each time
  third = lambda x: (duration * 2) // 3
  duration = third(1 + duration / 2)

  state = State(G, t, shared_prev_states, weight_function,
                butch_size=butch_size)

  state, temp = run(100, duration, state)

  while duration >= 1:
    duration, batch_size = third(duration), butch_size  #  half(butch_size)
    state, temp = run(temp, duration,
                      State(G, t, shared_prev_states,
                            weight_function,
                            S=state.S,
                            butch_size=butch_size),
                      )
  return state


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
    if not self.bootstrap(): return np.random.randint(1, self.butch_size)
    middle = (self.a + self.b) // 2
    return max(1, np.abs(s_size - middle))

  def bootstrap(self):
    return self.b != self.a


class State:
  def __init__(self, G, t, shared_prev_states,
               weight_func=None, S=None,
               _data=None, butch_size=1):
    self.G, self.t = G, t
    self.weight_func = weight_func
    self.shared_prev_states = shared_prev_states

    self.S = MySet(S) if S is not None else MySet()
    shared_prev_states.add(self.S)

    self.I = contagion_brd(self.G, self.S, self.t)
    self.I_comp = [i for i in range(self.G.n) if i not in self.I]
    self.data = self._calc_data(_data, butch_size)

  def is_cascading(self):
    return len(self.I_comp) == 0

  def _calc_data(self, data, butch_size: int):
    if data is None:
      return Data(0, self.G.n, butch_size)

    if not data.bootstrap():
      return data

    if self.is_cascading():
      return Data(data.a, len(self.S), data.butch_size)
    else:
      return Data(len(self.S), data.b, data.butch_size)

  def new_state(self, S=None, _new_game=True):
    return State(self.G, self.t, self.shared_prev_states, self.weight_func,
                 S, _data=None if _new_game else self.data)

  @property
  def S_weights(self):
    if self.weight_func is None: return None
    comp = self.I_comp if self.I_comp else self.S_comp
    return self.weight_func(self.S, comp, add=False)

  @property
  def C_weights(self):
    if self.weight_func is None: return None
    return self.weight_func(self.I_comp, self.I_comp, add=True)

  @property
  def S_comp(self):
    return [i for i in range(self.G.n) if i not in self.S]

  def __repr__(self):
    return f'Cascading={self.is_cascading()}_sLen={len(self.S)}'


def _accept_func(state, delta_l: float, Temp: float) -> object:
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


def _simulated_annealing(Temp: float, initial_state: State, alpha: float,
                         duration: int, loss: callable, accept: callable,
                         update: callable) -> tuple[State, float]:
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
  from helper_functions import get_out_degrees_weight_function
  G = create_fb_graph()
  t = 0.4
  weight_function = get_out_degrees_weight_function(G, exp=True)
  s = run_optimizer(G, t, duration=80, weight_function=weight_function)
  # None(80): 844
  # With(80): 814
  print(s)


if __name__ == '__main__':
  _test()
