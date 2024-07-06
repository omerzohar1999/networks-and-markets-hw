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
        return list(self.edges[nodeA])

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
        i, j = list(map(int, line.strip().split(" ")))
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
    # TODO: Implement this method
    pass


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
    # === Problem 9(b) === #
    # TODO: Put analysis code here
    # === Problem 9(c) === #
    # TODO: Put analysis code here
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
