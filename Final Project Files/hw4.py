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
        pass

    def add_edge(self, origin_node, destination_node):
        """Adds an edge from origin_node to destination_node."""
        pass

    def edges_from(self, origin_node):
        """This method shold return a list of all the nodes destination_node such that there is
        a directed edge (origin_node, destination_node) in the graph."""
        pass

    def get_edge(self, origin_node, destination_node):
        """This method should return true is there is an edge from origin_node to destination_node
        and false otherwise"""
        pass

    def number_of_nodes(self):
        """This method should return the number of nodes in the graph"""
        pass


# === Problem 7. ===
def scaled_page_rank(G, num_iter, eps=1 / 7.0):
    """This method, given a DirectedGraph G, runs the epsilon-scaled
    page-rank algorithm for num-iter iterations, for parameter eps,
    and returns a Dictionary where the keys are the set of
    nodes [0,...,G.number_of_nodes() - 1], each associated with a value
    equal to the score of output by the eps-scaled pagerank algorithm.

    In the case of num_iter=0, all nodes should
    have weight 1/G.number_of_nodes()"""
    pass


def graph_15_1_left():
    """This method, should construct and return a DirectedGraph encoding the left example in fig 15.1
    Use the following indexes: A:0, B:1, C:2, Z:3"""
    pass


def graph_15_1_right():
    """This method, should construct and return a DirectedGraph encoding the right example in fig 15.1
    Use the following indexes: A:0, B:1, C:2, Z1:3, Z2:4"""
    pass


def graph_15_2():
    """This method, should construct and return a DirectedGraph encoding example 15.2
    Use the following indexes: A:0, B:1, C:2, A':3, B':4, C':5"""
    pass


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
    pass


def main():
    # TODO: Put your analysis and plotting code here for 8(b)
    print("hello world")


if __name__ == "__main__":
    main()
