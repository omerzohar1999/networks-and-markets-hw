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
        if origin_node not in self.edges:
            return []
        return list(self.edges[origin_node].keys())

    def get_edge(self, origin_node, destination_node):
        """This method should return the weight (an integer > 0)
        if there is an edge between origin_node and
        destination_node, and 0 otherwise."""
        if origin_node not in self.edges:
            return 0
        return self.edges[origin_node].get(destination_node, 0)

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
        for origin in range(G.number_of_nodes()):
            for destination in G.edges_from(origin):
                capacity = G.get_edge(origin, destination)
                flow = F.get_edge(origin, destination)
                if flow < capacity:
                    R.set_edge(origin, destination, capacity - flow)
                if flow > 0:
                    R.set_edge(destination, origin, flow)
        return R

    def path_flow(G, path):
        flow = float("inf")
        for i in range(len(path) - 1):
            flow = min(flow, G.get_edge(path[i], path[i + 1]))
        return flow

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
            F.set_edge(origin, destination, F.get_edge(origin, destination) + flow)
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
    assert max_flow_fig_6_1[0] == 2

    max_flow_fig_6_3 = q_matching_fig_6_3()
    assert max_flow_fig_6_3[0] == 4


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
        for j in range(m):
            if C[i][j]:
                match_graph.set_edge(i + 1, n + j + 1, 1)
    for j in range(m):
        match_graph.set_edge(n + j + 1, sink, 1)
    _, F = max_flow(match_graph, source, sink)
    M = [None] * n
    for i in range(n):
        for j in range(m):
            if F.get_edge(i + 1, n + j + 1) > 0:
                M[i] = j
                break
    return M


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
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            C[i][j] = int(np.random.rand() < p)
    return C


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
        for _ in range(num_iters):
            C = random_driver_rider_bipartite_graph(n, p)
            result = max_matching(n, n, C)
            times_was_fully_matched += None not in result
        avg_fully_matched = times_was_fully_matched / num_iters
        print(f"p={p}, estimated fully matched p={avg_fully_matched}.")
        results.append(avg_fully_matched)
    plt.plot(ps, results)
    plt.xlabel("p")
    plt.ylabel("Estimated probability of full matching")
    plt.title("Estimated probability of full matching vs. p")
    plt.show()


if __name__ == "__main__":
    main()
