# Skeleton file for HW1
# =====================================
# IMPORTANT: You are NOT allowed to modify the method signatures
# (i.e. the arguments and return types each function takes).
# We will pass your grade through an autograder which expects a specific format.
# =====================================


# Do not include any other files or an external package, unless it is one of
# [numpy, pandas, scipy, matplotlib, random]
# please contact us before sumission if you want another package approved.
import numpy as np
import random
import matplotlib.pyplot as plt
from queue import Queue


# Implement the methods in this class as appropriate. Feel free to add other methods
# and attributes as needed.
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


# Problem 9(a)
def create_graph(n, p):
    """Given number of nodes n and probability p, output an UndirectedGraph with n nodes, where each
    pair of nodes is connected by an edge with probability p"""
    graph = UndirectedGraph(n)
    edges = np.random.rand(n, n) < p
    for i in range(n):
        for j in range(i):
            if edges[i][j]:
                graph.add_edge(i, j)
    return graph


# Problem 9(b)
def shortest_path(G, i, j):
    """Given an UndirectedGraph G and nodes i,j, output the length of the shortest path between i and j in G.
    If i and j are disconnected, output -1."""
    return G.get_distance(i, j)


# Problem 9(c)
def avg_shortest_path(G, num_samples=1000):
    """Given an UndirectedGraph G, return an estimate of the average shortest path in G, where the average is taken
    over all pairs of CONNECTED nodes. The estimate should be taken by sampling num_samples random pairs of connected nodes,
    and computing the average of their shortest paths. Return a decimal number."""
    dists = 0.0
    n = G.number_of_nodes()
    for _ in range(num_samples):
        i, j = random.sample(range(n), 2)
        dists += G.get_distance(i, j)
    return dists / num_samples


# Problem 10(a)
def create_fb_graph(filename="facebook_combined.txt"):
    """This method should return a undirected version of the facebook graph as an instance of the UndirectedGraph class.
    You may assume that the input graph has 4039 nodes."""
    num_nodes = 4039
    graph = UndirectedGraph(num_nodes)
    for line in open(filename):
        i, j = list(map(int, line.strip().split(" ")))
        graph.add_edge(i, j)
    return graph


def random_graph_average_distance(n, p):
    graph = create_graph(n, p)
    while not graph.is_connected():
        graph = create_graph(n, p)
    return avg_shortest_path(graph)


def Q9d():
    ps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    average_distance = []
    theoretical_distance = []
    n = 1000
    for p in ps:
        average_distance.append(random_graph_average_distance(n, p))
        print(f"average distance for graph({n=},{p=}) is {average_distance[-1]}")
        theoretical_distance.append(2 - p)
    plt.plot(ps, average_distance, label="estimated distance")
    plt.plot(ps, theoretical_distance, label="the function f(p)=2-p")
    plt.xlabel("probability for edge between each pair of nodes")
    plt.xlabel("estimate for average distance")
    plt.legend()
    plt.show()


def Q10b(fb_graph):
    print(
        f"average distance estimate for facebook graph is {avg_shortest_path(fb_graph)}"
    )


def Q10c(fb_graph: UndirectedGraph):
    n = fb_graph.n
    edges = fb_graph.edges
    num_pairs = (n * (n - 1)) // 2
    num_edges = sum(map(len, edges.values())) // 2
    estimated_p = num_edges / num_pairs
    print(f"Estimated p for facebook graph is {estimated_p}")
    return estimated_p


def main():
    # Q9:
    Q9d()
    # Q10:
    fb_graph = create_fb_graph()
    print(f"{fb_graph.n=}")
    Q10b(fb_graph)
    estimated_p = Q10c(fb_graph)
    random_graph_with_p_dist = random_graph_average_distance(fb_graph.n, estimated_p)
    print(
        f"If the Facebook graph was random with same n,p,"
        f"we'd expect it to have an average distance of {random_graph_with_p_dist}"
    )


if __name__ == "__main__":
    main()
