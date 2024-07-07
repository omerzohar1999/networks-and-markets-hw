from local_search import run_optimizer, State, loss_func
import multiprocessing as mp


def run_annealing_parallel(G, t, duration: int,
                           add_func: callable = None,
                           remove_func: callable = None,
                           num_processes: int = 4) -> State:
  if num_processes == 1:
    return _process_thread_pool(G, t, duration, add_func, remove_func)

  with mp.Pool(processes=num_processes) as pool:
    process_results = pool.starmap(
      _process_thread_pool,
      [(G, t, duration, add_func, remove_func) for _ in range(num_processes)]
    )

  return min(process_results, key=lambda x: loss_func(x))


def _process_thread_pool(G, t, duration: int,
                         add_func: callable = None,
                         remove_func: callable = None) -> State:
  return run_optimizer(
    G, t, duration, add_func, remove_func
  )
