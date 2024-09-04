# Skeleton file for HW4 question 4
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


# Implement the methods in this class as appropriate. Feel free to add other methods
# and attributes as needed. You may/should reuse code from previous HWs when applicable.
class DirectedGraph:
    def __init__(self, number_of_nodes):
        """Assume that nodes are represented by indices/integers between 0 and number_of_nodes - 1."""
        self.n = number_of_nodes
        self.edges = dict()

    def add_edge(self, origin_node, destination_node):
        """Adds an edge from origin_node to destination_node."""
        if origin_node not in self.edges:
            self.edges[origin_node] = set()
        self.edges[origin_node].add(destination_node)

    def edges_from(self, origin_node):
        """This method shold return a list of all the nodes destination_node such that there is
        a directed edge (origin_node, destination_node) in the graph."""
        return list(self.edges.get(origin_node, set()))

    def get_edge(self, origin_node, destination_node):
        """This method should return true is there is an edge from origin_node to destination_node
        and false otherwise"""
        return destination_node in self.edges.get(origin_node, set())

    def number_of_nodes(self):
        """This method should return the number of nodes in the graph"""
        return self.n


# === Problem 7. ===
def scaled_page_rank(G: DirectedGraph, num_iter: int, eps: int = 1 / 7.0):
    """This method, given a DirectedGraph G, runs the epsilon-scaled
    page-rank algorithm for num-iter iterations, for parameter eps,
    and returns a Dictionary where the keys are the set of
    nodes [0,...,G.number_of_nodes() - 1], each associated with a value
    equal to the score of output by the eps-scaled pagerank algorithm.

    In the case of num_iter=0, all nodes should
    have weight 1/G.number_of_nodes()"""
    weights = [1 / G.number_of_nodes()] * G.number_of_nodes()
    for _ in range(num_iter):
        new_weights = [eps / G.number_of_nodes()] * G.number_of_nodes()
        for i in range(G.number_of_nodes()):
            for j in G.edges_from(i):
                new_weights[j] += (1 - eps) * weights[i] / len(G.edges_from(i))
        weights = new_weights
    return {i: weights[i] for i in range(G.number_of_nodes())}


def graph_15_1_left():
    """This method, should construct and return a DirectedGraph encoding the left example in fig 15.1
    Use the following indexes: A:0, B:1, C:2, Z:3"""
    G = DirectedGraph(4)
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 0)
    G.add_edge(0, 3)
    G.add_edge(3, 3)

    return G


def graph_15_1_right():
    """This method, should construct and return a DirectedGraph encoding the right example in fig 15.1
    Use the following indexes: A:0, B:1, C:2, Z1:3, Z2:4"""
    G = DirectedGraph(5)

    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 0)
    G.add_edge(0, 3)
    G.add_edge(3, 4)
    G.add_edge(0, 4)
    G.add_edge(4, 3)

    return G


def graph_15_2():
    """This method, should construct and return a DirectedGraph encoding example 15.2
    Use the following indexes: A:0, B:1, C:2, A':3, B':4, C':5"""
    G = DirectedGraph(6)

    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 0)

    G.add_edge(3, 4)
    G.add_edge(4, 5)
    G.add_edge(5, 3)

    return G


def extra_graph_1():
    """This method, should construct and return a DirectedGraph of your choice with at least 10 nodes"""
    pass


def extra_graph_2():
    """This method, should construct and return a DirectedGraph of your choice with at least 10 nodes"""
    pass


# === Problem 8. ===
def facebook_graph(filename="facebook_combined.txt"):
    """This method should return a DIRECTED version of the facebook graph as an instance of the DirectedGraph class.
    In particular, if u and v are friends, there should be an edge between u and v and an edge between v and u.
    """
    num_nodes = 4039
    graph = DirectedGraph(num_nodes)
    for line in open(filename):
        i, j = list(map(int, line.strip().split(" ")))
        graph.add_edge(i, j)
        graph.add_edge(j, i)
    return graph


def main():
    fb_graph = facebook_graph()
    ranks = scaled_page_rank(fb_graph, 25)
    # TODO: 8 (c), (d)


if __name__ == "__main__":
    main()
