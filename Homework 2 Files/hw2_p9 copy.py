# Skeleton file for HW2 question 9
# =====================================
# IMPORTANT: You are NOT allowed to modify the method signatures
# (i.e. the arguments and return types each function takes).
# We will pass your grade through an autograder which expects a specific format.
# =====================================


# Do not include any other files or an external package, unless it is one of
# [numpy, pandas, scipy, matplotlib, random]
# please contact us before sumission if you want another package approved.
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
import random
from itertools import product



class UndirectedGraph:
    def __init__(self, A):
        self.n = len(A)
        self.A = np.array(A, dtype=bool)
        self.D = None

    def add_edge(self, i, j):
        if i != j:
            self.A[i,j] = self.A[j,i] = True
        self.D = None

    def neighbors(self, nodeA):
        return np.where(self.A[nodeA])[0]

    def check_edge(self, i, j):
        return self.A[i,j] == 1

    def _apd(self):
        A, n = self.A, self.n

        if self.D is not None:
            return
        if not self.A:
            self.D = np.array([[]])
            return

        D = np.where(A != 0, A, np.inf)
        np.fill_diagonal(A, 0)  # Set diagonal to 0

        for k in range(n):
            for i, j in product(range(n), repeat=2):
                D[i, j] = min(D[i, j], D[i, k] + D[k, j])

        self.D = D

    def get_distance(self, i, j):
        if self.D is None:
            self._apd()
        return self.D[i,j]

    def is_connected(self):
        if self.D is None:
            self._apd()
        return np.all(self.D != np.inf)

def adjacency_matrix_from_edges(edges, nodes=4039):
    n = nodes
    A = np.zeros((n, n), dtype=bool)
    for i, j in edges:
        A[i,j] = A[j,i] = True
    return A

def create_fb_graph(filename="facebook_combined.txt"):
    """This method should return a undirected version of the facebook graph as an instance of the UndirectedGraph class.
    You may assume that the input graph has 4039 nodes."""
    edges = (map(int, line.strip().split())
             for line in open(filename))
    A = adjacency_matrix_from_edges(edges)
    return UndirectedGraph(A)


# === Problem 9(a) ===


def contagion_brd(G, S, t):
    """Given an UndirectedGraph G, a list of adopters S (a list of integers in [0, G.number_of_nodes - 1]),
    and a float threshold t, perform BRD as follows:
    - Permanently infect the nodes in S with X
    - Infect the rest of the nodes with Y
    - Run BRD on the set of nodes not in S
    Return a list of all nodes infected with X after BRD converges."""
    S = set(S)
    is_x = [(idx in S) for idx in range(G.n)]
    needs_addressing = {idx for idx in range(G.n) if idx not in S}

    def brd_step(i):
        neighbors = G.neighbors(i)
        X_neighbors = [n for n in neighbors if is_x[n]]
        frac = len(X_neighbors) / len(neighbors)

        neighbors_not_in_s = lambda: \
            {n for n in neighbors if n not in S}

        if frac > t and not is_x[i]:
            is_x[i] = True
            needs_addressing.update(neighbors_not_in_s())

        if frac < t and is_x[i]:
            is_x[i] = False
            needs_addressing.update(neighbors_not_in_s())

    while needs_addressing:
        brd_step(needs_addressing.pop())

    return [i for i in range(G.n) if is_x[i]]


def q_completecascade_graph_fig4_1_left():
    """Return a float t s.t. the left graph in Figure 4.1 cascades completely."""
    # TODO: Implement this method
    pass


def q_incompletecascade_graph_fig4_1_left():
    """Return a float t s.t. the left graph in Figure 4.1 does not cascade completely."""
    # TODO: Implement this method
    pass


def q_completecascade_graph_fig4_1_right():
    """Return a float t s.t. the right graph in Figure 4.1 cascades completely."""
    # TODO: Implement this method
    pass


def q_incompletecascade_graph_fig4_1_right():
    """Return a float t s.t. the right graph in Figure 4.1 does not cascade completely."""
    # TODO: Implement this method
    pass


def main():
    fb_graph = create_fb_graph()
    # === Problem 9(b) === #
    threshold = 0.1
    k = 10
    T = 100
    num_infected_total = 0
    times_infected = 0
    for i in range(T):
        S = random.sample(range(fb_graph.n), k)
        infected = contagion_brd(fb_graph, S, threshold)
        num_infected = len(infected)
        num_infected_total += num_infected
        if num_infected == fb_graph.n:
            times_infected += 1
            print(f"graph was fully infected on iteration {i}")
    avg_num_infected = num_infected_total / T
    print(f"{avg_num_infected=}")
    print(f"infected {times_infected}/{T} times.")
    # === Problem 9(c) === #
    ts = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    ks = list(range(0, 251, 10))
    T = 10
    for threshold in ts:
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
            print(
                f"{threshold=}, {k=}, {avg_num_infected=}, fully infected {times_infected}/{T} times."
            )
    # === OPTIONAL: Bonus Question 2 === #
    # TODO: Put analysis code here
    pass


# === OPTIONAL: Bonus Question 2 === #
def min_early_adopters(G, q):
    """Given an undirected graph G, and float threshold t, approximate the
    smallest number of early adopters that will call a complete cascade.
    Return an integer between [0, G.number_of_nodes()]"""
    pass


if __name__ == "__main__":
    main()



