# Skeleton file for HW2 question 10
# =====================================
# IMPORTANT: You are NOT allowed to modify the method signatures
# (i.e. the arguments and return types each function takes).
# We will pass your grade through an autograder which expects a specific format.
# =====================================


# Do not include any other files or an external package, unless it is one of
# [numpy, pandas, scipy, matplotlib, random]
# and UndirectGraph from hw2_p9
# please contact us before sumission if you want another package approved.
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
from hw2_p9 import UndirectedGraph


# Implement the methods in this class as appropriate. Feel free to add other methods
# and attributes as needed. You may/should reuse code from previous HWs when applicable.
class WeightedDirectedGraph:
    def __init__(self, number_of_nodes):
        """Assume that nodes are represented by indices/integers between 0 and number_of_nodes - 1."""
        self.n = number_of_nodes
        self.edges = {node: dict() for node in range(number_of_nodes)}
        self.distances = {}

    def set_edge(self, origin_node, destination_node, weight=1):
        """Modifies the weight for the specified directed edge, from origin to destination node,
        with specified weight (an integer >= 0). If weight = 0, effectively removes the edge from
        the graph. If edge previously wasn't in the graph, adds a new edge with specified weight.
        """
        if weight > 0:
            self.edges[origin_node][destination_node] = weight
        elif destination_node in self.edges[origin_node]:
            self.edges[origin_node].pop(destination_node)
        self.distances.clear()

    def edges_from(self, origin_node):
        """This method shold return a list of all the nodes destination_node such that there is
        a directed edge (origin_node, destination_node) in the graph (i.e. with weight > 0).
        """
        return list(self.edges[origin_node].keys()) \
            if origin_node in self.edges else []
        # if origin_node not in self.edges:
        #     return []
        # return list(self.edges[origin_node].keys())

    def get_edge(self, origin_node, destination_node):
        """This method should return the weight (an integer > 0)
        if there is an edge between origin_node and
        destination_node, and 0 otherwise."""
        return self.edges[origin_node].get(destination_node, 0) \
            if origin_node in self.edges else 0
        # if origin_node not in self.edges:
        #     return 0
        # return self.edges[origin_node].get(destination_node, 0)

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
            if popped not in self.edges:
                continue
            for neighbor in self.edges[popped]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[popped] + 1
                    parents[neighbor] = popped
                    nodes_to_visit.put(neighbor)

        self.distances[sNode] = distances

    def get_path(self, s, t):
        visited = set()
        distances = [-1] * self.n
        parents = distances.copy()

        visited.add(s)
        distances[s] = 0

        nodes_to_visit = Queue()

        nodes_to_visit.put(s)

        while nodes_to_visit.qsize():
            popped = nodes_to_visit.get()
            if popped == t:
                break
            if popped not in self.edges:
                continue
            for neighbor in self.edges[popped]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[popped] + 1
                    parents[neighbor] = popped
                    nodes_to_visit.put(neighbor)
        if distances[t] == -1:
            return None
        path = [t]
        while path[-1] != s:
            path.append(parents[path[-1]])
        return path[::-1]


# === Problem 10(a) ===
def max_flow(G, s, t):
    """Given a WeightedDirectedGraph G, a source node s, a destination node t,
    compute the (integer) maximum flow from s to t, treating the weights of G as capacities.
    Return a tuple (v, F) where v is the integer value of the flow, and F is a maximum flow
    for G, represented by another WeightedDirectedGraph where edge weights represent
    the final allocated flow along that edge."""

    def residual_graph(G, F):
        R = WeightedDirectedGraph(G.number_of_nodes())

        # initialize R to be the same as G (capacity graph)
        for origin in range(G.number_of_nodes()):
            for destination in G.edges_from(origin):
                R.set_edge(origin, destination, G.get_edge(origin, destination))

        # update R with the flow graph F
        for origin in range(G.number_of_nodes()):
            for destination in G.edges_from(origin):
                flow = F.get_edge(origin, destination)
                R.set_edge(origin, destination, R.get_edge(origin, destination) - flow)
                R.set_edge(destination, origin, R.get_edge(destination, origin) + flow)

        # for origin in range(G.number_of_nodes()):
        #     for destination in G.edges_from(origin):
        #         capacity = G.get_edge(origin, destination)
        #         flow = F.get_edge(origin, destination)
        #         if flow < capacity:
        #             R.set_edge(origin, destination, capacity - flow)
        #         if flow > 0:
        #             R.set_edge(destination, origin, flow)

        return R

    def path_flow(G, path):
        return min(G.get_edge(path[i], path[i + 1])
                   for i in range(len(path) - 1))
        # flow = float("inf")
        # for i in range(len(path) - 1):
        #     flow = min(flow, G.get_edge(path[i], path[i + 1]))
        # return flow

    # init F
    F = WeightedDirectedGraph(G.number_of_nodes())
    for origin in range(G.number_of_nodes()):
        for destination in G.edges_from(origin):
            F.set_edge(origin, destination, 0)

    # find augmenting paths, update F
    residual = residual_graph(G, F)
    path = residual.get_path(s, t)
    while path is not None:
        flow = path_flow(residual, path)
        for i in range(len(path) - 1):
            origin = path[i]
            destination = path[i + 1]

            # calc the forward and backward flow additions (NOTE: I think this is true, but I'm not sure)
            forward_flow = min(flow, G.get_edge(origin, destination) - F.get_edge(origin, destination))
            residual_flow = flow - forward_flow
            
            # update the flow graph
            F.set_edge(origin, destination, F.get_edge(origin, destination) + forward_flow)
            F.set_edge(destination, origin, F.get_edge(destination, origin) - residual_flow)
            
        residual = residual_graph(G, F)
        path = residual.get_path(s, t)

    # calculate v
    v = 0
    for neighbor in F.edges_from(s):
        v += F.get_edge(s, neighbor)
    return v, F


def q_matching_fig_6_1():
    G = WeightedDirectedGraph(4)
    G.set_edge(0, 1, 1)
    G.set_edge(0, 2, 3)
    G.set_edge(1, 2, 2)
    G.set_edge(1, 3, 1)
    G.set_edge(2, 3, 1)
    return max_flow(G, 0, 3)


def q_matching_fig_6_3():
    G = WeightedDirectedGraph(12)
    G.set_edge(0, 1, 1)
    G.set_edge(0, 2, 1)
    G.set_edge(0, 3, 1)
    G.set_edge(0, 4, 1)
    G.set_edge(0, 5, 1)
    G.set_edge(1, 7, 1)
    G.set_edge(2, 6, 1)
    G.set_edge(2, 7, 1)
    G.set_edge(3, 6, 1)
    G.set_edge(4, 8, 1)
    G.set_edge(5, 8, 1)
    G.set_edge(5, 9, 1)
    G.set_edge(4, 10, 1)
    G.set_edge(6, 11, 1)
    G.set_edge(7, 11, 1)
    G.set_edge(8, 11, 1)
    G.set_edge(9, 11, 1)
    G.set_edge(10, 11, 1)

    return max_flow(G, 0, 11)


def max_flow_sanity_checks():
    max_flow_fig_6_1 = q_matching_fig_6_1()
    assert max_flow_fig_6_1[0] == 2, "Failure: Expected 2, got " + str(max_flow_fig_6_1[0])

    max_flow_fig_6_3 = q_matching_fig_6_3()
    assert max_flow_fig_6_3[0] == 4, "Failure: Expected 4, got " + str(max_flow_fig_6_3[0])


# === Problem 10(c) ===
def max_matching(n, m, C):
    """Given n drivers, m riders, and a set of matching constraints C,
    output a maximum matching. Specifically, C is a n x m array, where
    C[i][j] = 1 if driver i (in 0...n-1) and rider j (in 0...m-1) are compatible.
    If driver i and rider j are incompatible, then C[i][j] = 0.
    Return an n-element array M where M[i] = j if driver i is matched with rider j,
    and M[i] = None if driver i is not matched."""
    match_graph = WeightedDirectedGraph(n + m + 2)
    source = 0
    sink = n + m + 1
    for i in range(n):
        match_graph.set_edge(source, i + 1, 1)
        for j in filter(lambda j: C[i][j], range(m)):
            match_graph.set_edge(i + 1, n + j + 1, 1)
    for j in range(m):
        match_graph.set_edge(n + j + 1, sink, 1)
    flow_val, F = max_flow(match_graph, source, sink)

    # Extract the matching
    matching = []
    for i in range(n):
        matching.append(None)
        for j in range(m):
            if F.get_edge(i + 1, n + j + 1) == 1:
                matching[i] = j
                break

    # Return the matching
    return matching


def max_matching_sanity_checks():
    C1 = [
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ]
    result = max_matching(5, 5, C1)
    assert len(result) == 5
    assert result == [0, None, None, None, None]

    C2 = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
    result = max_matching(5, 5, C2)
    assert len(result) == 5
    assert result == [0, 1, 2, 3, 4]

    C3 = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    result = max_matching(5, 5, C3)
    assert len(result) == 5
    assert None not in result
    assert len(set(result)) == 5

    C4 = [
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1],
    ]
    result = max_matching(5, 5, C4)
    assert len(result) == 5
    assert None not in result
    assert len(set(result)) == 5
    assert result[0] in {0, 1}
    assert result[1] in {1, 2}
    assert result[2] in {2, 3}
    assert result[3] in {3, 4}
    assert result[4] in {4, 0}


# === Problem 10(d) ===
def random_driver_rider_bipartite_graph(n, p):
    """Returns an n x n constraints array C as defined for max_matching, representing a bipartite
    graph with 2n nodes, where each vertex in the left half is connected to any given vertex in the
    right half with probability p."""
    return list(map(list, (np.random.rand(n, n) < p).astype(int)))

    # from itertools import product
    # C = [[0] * n for _ in range(n)]

    # for i, j in product(range(n), repeat=2):
    #     C[i][j] = int(np.random.rand() < p)

    # for i in range(n):
    #     for j in range(n):
    #         C[i][j] = int(np.random.rand() < p)
    # return C


# === Bonus Question 3 ===
def get_min_p_for_full_matching(n, num_iters=30):
    """Given n drivers and n riders, estimate the minimum value of p such that the probability
    of a full matching is at least 0.99. Perform num_iters trials to estimate this probability.
    We do so by exponentially increasing p until the estimated probability of full matching is, and then binary searching for the exact value in the range [p / 2, p]."""
    p = 1
    while True:
        print(f"p={p}")
        times_was_fully_matched = 0
        for j in range(num_iters):
            C = random_driver_rider_bipartite_graph(n, p)
            result = max_matching(n, n, C)
            times_was_fully_matched += None not in result
        avg_fully_matched = times_was_fully_matched / num_iters
        if avg_fully_matched < 0.99:
            break
        p /= 2
    lower, upper = p, 2 * p
    while upper - lower > 1e-3:
        print(f"lower={lower}, upper={upper}")
        p = (lower + upper) / 2
        times_was_fully_matched = 0
        for j in range(num_iters):
            C = random_driver_rider_bipartite_graph(n, p)
            result = max_matching(n, n, C)
            times_was_fully_matched += None not in result
        avg_fully_matched = times_was_fully_matched / num_iters
        if avg_fully_matched >= 0.99:
            upper = p
        else:
            lower = p
    return p

def plot_min_p_for_full_matching():
    ns = range(10, 100, 5)
    min_ps = []
    for n in ns:
        min_ps.append(get_min_p_for_full_matching(n, num_iters=int(1000 / n)))
        print(f"n={n}, p={min_ps[-1]}")
    plt.plot(ns, min_ps, label="Estimated minimum p for full matching")
    plt.xlabel("n")
    plt.ylabel("Minimum p for full matching")
    plt.title("Minimum p for full matching vs. n")

    # add plot of approximation function
    x = np.linspace(min(ns), max(ns), 100)
    y = 1 / np.sqrt(x)
    plt.plot(x, y, label="1 / sqrt(n)")
    plt.legend()
    # plt.savefig("bonus3.png", format="png")
    # plt.savefig("bonus3.pgf", format="pgf")
    plt.show()

def main():
    max_flow_sanity_checks()
    max_matching_sanity_checks()
    # === Problem 10(d) === #
    print("\nQ10d\n")
    n = 100
    num_iters = 100
    ps = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    results = []
    for p in ps:
        times_was_fully_matched = 0
        for j in range(num_iters):
            print(f"\riter={j+1} out of {num_iters}", end="")
            C = random_driver_rider_bipartite_graph(n, p)
            result = max_matching(n, n, C)
            times_was_fully_matched += None not in result
        avg_fully_matched = times_was_fully_matched / num_iters
        print(f"\rp={p}, estimated fully matched p={avg_fully_matched}.")
        results.append(avg_fully_matched)
    plt.plot(ps, results)
    plt.xlabel("p")
    plt.ylabel("Estimated probability of full matching")
    plt.title(f"Estimated probability of full matching vs. p ({n=})")
    # plt.savefig("q10d.png", format="png")
    # plt.savefig("q10d.pgf", format="pgf")
    plt.show()

    # === Bonus Question 3 === #
    plot_min_p_for_full_matching()


if __name__ == "__main__":
    main()
