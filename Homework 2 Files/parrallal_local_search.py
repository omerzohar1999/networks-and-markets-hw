import time

from local_search import run_optimizer, State, loss_func
import pathos.multiprocessing as mp


def run_annealing_parallel(G, t, duration: int,
                           weight_function: callable = None,
                           num_processes: int = 4) -> State:
  if num_processes == 1:
    return _process_thread_pool(G, t, duration, weight_function)

  with mp.Pool(processes=num_processes) as pool:
    process_results = pool.starmap(
      _process_thread_pool,
      [(G, t, duration, weight_function) for _ in range(num_processes)]
    )

  return min(process_results, key=lambda x: loss_func(x))


def _process_thread_pool(G, t, duration: int,
                         weight_function: callable = None) -> State:
  return run_optimizer(
    G, t, duration, weight_function
  )


def _test():
  from hw2_p9 import create_fb_graph
  from helper_functions import get_out_degrees_weight_function
  G = create_fb_graph()
  t = 0.4
  weight_function = get_out_degrees_weight_function(G, exp=True)
  s = run_annealing_parallel(G, t,
                             duration=80,
                             weight_function=weight_function,
                             num_processes=8)
  # with(60): 866
  # with(80): 810, 852
  print(s)


if __name__ == '__main__':
  _test()
