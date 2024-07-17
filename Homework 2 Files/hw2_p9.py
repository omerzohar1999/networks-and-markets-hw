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


    # === OPTIONAL: Bonus Question 2 === #
    # TODO: Put analysis code here
    from local_search import run_optimizer_par
    min_s_size = []
    thresholds = np.linspace(0.1, 0.9, 10)
    for t in thresholds:
        print(f'\r running optimizer for t = {t:.2f}..', end='')
        len_s = len(run_optimizer_par(fb_graph, t, duration=10))
        print(f'\r running optimizer for t = {t:.2f}.. size of S = {len_s}')

        min_s_size.append(len_s)

    plt.plot(thresholds, min_s_size)
    plt.xlabel('Threshold (t)')
    plt.ylabel('Minimal size of S')
    plt.title('Approximating minimal S size per threshold')
    plt.grid(False)
    plt.show()



# === OPTIONAL: Bonus Question 2 === #


def min_early_adopters(G, q):
    """Given an undirected graph G, and float threshold t, approximate the
    smallest number of early adopters that will call a complete cascade.
    Return an integer between [0, G.number_of_nodes()]"""
    pass


if __name__ == "__main__":
    main()
