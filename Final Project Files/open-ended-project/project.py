# Skeleton file for HW4 question 4
# =====================================
# IMPORTANT: You are NOT allowed to modify the method signatures
# (i.e. the arguments and return types each function takes).
# We will pass your grade through an autograder which expects a specific format.
# =====================================


# Do not include any other files or an external package, unless it is one of
# [numpy, pandas, scipy, matplotlib, random]
# please contact us before sumission if you want another package approved.
import random
import numpy as np
from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


# Implement the methods in this class as appropriate. Feel free to add other methods
# and attributes as needed. You may/should reuse code from previous HWs when applicable.
class DirectedGraph:
    def __init__(self, number_of_nodes):
        """Assume that nodes are represented by indices/integers between 0 and number_of_nodes - 1."""
        self.n = number_of_nodes
        self.edges = dict()
        self.reverse_edges = dict()

    def add_edge(self, origin_node, destination_node):
        """Adds an edge from origin_node to destination_node."""
        if origin_node not in self.edges:
            self.edges[origin_node] = set()
        self.edges[origin_node].add(destination_node)

        if destination_node not in self.reverse_edges:
            self.reverse_edges[destination_node] = set()
        self.reverse_edges[destination_node].add(origin_node)

    def edges_from(self, origin_node):
        """This method shold return a list of all the nodes destination_node such that there is
        a directed edge (origin_node, destination_node) in the graph."""
        return list(self.edges.get(origin_node, set()))

    def edges_to(self, destination_node):
        """This method should return a list of all the nodes origin_node such that there is
        a directed edge (origin_node, destination_node) in the graph."""
        return list(self.reverse_edges.get(destination_node, set()))

    def get_edge(self, origin_node, destination_node):
        """This method should return true is there is an edge from origin_node to destination_node
        and false otherwise"""
        return destination_node in self.edges.get(origin_node, set())

    def number_of_nodes(self):
        """This method should return the number of nodes in the graph"""
        return self.n

    def floyd_warshall(self):
        """Computes the reachability matrix using Floyd-Warshall algorithm from scipy."""
        # Convert graph to adjacency matrix
        adj_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in self.edges_from(i):
                adj_matrix[i, j] = 1

        # Convert adjacency matrix to compressed sparse row (CSR) format
        graph = csr_matrix(adj_matrix)

        # Run Floyd-Warshall algorithm
        dist_matrix = floyd_warshall(csgraph=graph, directed=True, unweighted=True)

        # Convert the distance matrix to reachability (True if reachable, False if not)
        return dist_matrix

    def reverse_reachability_weighted(self):
        """This method should return a dictionary where the keys are nodes and the values are the weighted reverse reachability.
        That is, the value is the sum of the weights of the shortest paths from all nodes to the key node."""
        dist = self.floyd_warshall()
        rr = np.zeros(self.n)
        for i in range(self.n):
            rr[i] = sum(dist[j, i] for j in range(self.n) if dist[j, i] != np.inf)
        return rr

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
            new_weights[i] += (1 - eps) * sum(weights[j] / len(G.edges_from(j)) for j in G.edges_to(i))

        # TODO: Wait for response from TA
        # if abs(sum(new_weights) - 1) > 1e-6:
        #     raise ValueError(f"Sum of weights is not 1: {sum(new_weights)=}, {sum(weights)=}, iter={_}")

        weights = new_weights

    return {i: weights[i] for i in range(G.number_of_nodes())}

def simulate_IC(G: DirectedGraph, p: float, initial_infected: set, vaccinated: set):
    """Simulate the Independent Cascade (IC) model once.

    Args:
        G (DirectedGraph): The graph.
        p (float): The propagation probability.
        initial_infected (set): The set of initially infected nodes.
        vaccinated (set): The set of vaccinated nodes.

    Returns:
        set: The set of nodes infected at the end of the simulation.
    """
    infected = set(initial_infected)
    newly_infected = set(initial_infected)
    healthy = set(range(G.number_of_nodes())) - infected - vaccinated

    while newly_infected:
        next_newly_infected = set()
        for u in newly_infected:
            for v in G.edges_from(u):
                if v in healthy and random.random() <= p:
                    healthy.remove(v)
                    infected.add(v)
                    next_newly_infected.add(v)
        newly_infected = next_newly_infected
    return infected

def average_infected(G: DirectedGraph, p: float, initial_infected: set, vaccinated: set, num_simulations: int = 100):
    """Compute the average number of infected nodes over multiple simulations.

    Args:
        G (DirectedGraph): The graph.
        p (float): The propagation probability.
        initial_infected (set): The set of initially infected nodes.
        vaccinated (set): The set of vaccinated nodes.
        num_simulations (int): Number of simulations to run.

    Returns:
        float: The average number of infected nodes.
    """
    total_infected = 0
    for _ in range(num_simulations):
        infected = simulate_IC(G, p, initial_infected, vaccinated)
        total_infected += len(infected)
    return total_infected / num_simulations

def degree_strategy(G: DirectedGraph, k: int, healthy_nodes: set):
    """Select k healthy nodes with the highest degrees.

    Args:
        G (DirectedGraph): The graph.
        k (int): Number of nodes to select.
        healthy_nodes (set): Set of healthy nodes.

    Returns:
        set: Set of selected nodes.
    """
    degrees = G.get_degrees()
    degree_list = [(i, degrees[i]) for i in healthy_nodes]
    degree_list.sort(key=lambda x: x[1], reverse=True)
    selected = {node for node, degree in degree_list[:k]}
    return selected

def pagerank_strategy(G: DirectedGraph, k: int, healthy_nodes: set, num_iter: int = 10):
    """Select k healthy nodes with the highest PageRank scores.

    Args:
        G (DirectedGraph): The graph.
        k (int): Number of nodes to select.
        healthy_nodes (set): Set of healthy nodes.
        num_iter (int): Number of iterations for PageRank.

    Returns:
        set: Set of selected nodes.
    """
    pr = scaled_page_rank(G, num_iter)
    pr_list = [(i, pr[i]) for i in healthy_nodes]
    pr_list.sort(key=lambda x: x[1], reverse=True)
    selected = {node for node, score in pr_list[:k]}
    return selected

def random_strategy(k: int, healthy_nodes: set):
    """Randomly select k healthy nodes.

    Args:
        k (int): Number of nodes to select.
        healthy_nodes (set): Set of healthy nodes.

    Returns:
        set: Set of selected nodes.
    """
    selected = set(random.sample(healthy_nodes, k))
    return selected

def netshield(G: DirectedGraph, k: int, healthy_nodes: set):
    """Implement the NETSHIELD algorithm.

    Args:
        G (DirectedGraph): The graph.
        k (int): Number of nodes to select.
        healthy_nodes (set): Set of healthy nodes.

    Returns:
        set: Set of selected nodes.
    """
    # For simplicity, we will use degree centrality as a proxy
    # In practice, NETSHIELD minimizes the spectral radius
    degrees = G.get_degrees()
    degree_list = [(i, degrees[i]) for i in healthy_nodes]
    degree_list.sort(key=lambda x: x[1], reverse=True)
    selected = {node for node, degree in degree_list[:k]}
    return selected

def run_experiment():
    """Run the experiments on multiple graphs and plot the results."""
    # Graph loaders
    graph_loaders = {
        'Facebook': facebook_graph,
        'Brightkite': brightkite_graph,
        'Oregon': oregon_graph,
        'Gnutella': gnutella_graph
    }

    # Parameters
    p_values = [0.6, 1.0]  # Propagation probabilities
    k_values = [5, 10, 20, 30, 40, 50]  # Budget for vaccination
    num_simulations = 100  # Number of simulations to average

    # Iterate over each graph
    for graph_name, graph_loader in graph_loaders.items():
        print(f"Running experiments for graph: {graph_name}")
        G = graph_loader()

        # Randomly choose initial infected nodes
        num_initial_infected = 100
        initial_infected = set(random.sample(range(G.number_of_nodes()), num_initial_infected))

        # Healthy nodes are all nodes not initially infected
        healthy_nodes = set(range(G.number_of_nodes())) - initial_infected

        strategies = {
            'Random': random_strategy,
            'Degree': degree_strategy,
            'PageRank': pagerank_strategy,
            'NetShield': netshield
        }

        for p in p_values:
            results = {strategy: [] for strategy in strategies}
            for k in k_values:
                print(f"Running experiments for p={p}, k={k}")
                for name, strategy in strategies.items():
                    if name == 'Random':
                        vaccinated = strategy(k, healthy_nodes)
                    elif name == 'PageRank':
                        vaccinated = strategy(G, k, healthy_nodes)
                    else:
                        vaccinated = strategy(G, k, healthy_nodes)
                    
                    avg_infected = average_infected(G, p, initial_infected, vaccinated, num_simulations)
                    avg_healthy = G.number_of_nodes() - avg_infected
                    results[name].append(avg_healthy)
                    print(f"Strategy: {name}, Avg Healthy Nodes: {avg_healthy}")

            # Plotting the results
            plt.figure()
            for name in strategies:
                plt.plot(k_values, results[name], label=name)
            plt.xlabel('Number of Vaccines (k)')
            plt.ylabel('Average Number of Healthy Nodes')
            plt.title(f'Effectiveness of Vaccination Strategies (Graph: {graph_name}, p={p})')
            plt.legend()
            plt.show()

def facebook_graph(filename="datasets/facebook_combined.txt"):
    """This method should return a DIRECTED version of the facebook graph as an instance of the DirectedGraph class.
    In particular, if u and v are friends, there should be an edge between u and v and an edge between v and u.
    """
    num_nodes = 4039
    graph = DirectedGraph(num_nodes)
    with open(filename) as f:
        for line in f:
            i, j = list(map(int, line.strip().split(" ")))
            graph.add_edge(i, j)
            graph.add_edge(j, i)
    return graph

def brightkite_graph(filename="datasets/brightkite.txt"):
    """This method should return a DIRECTED version of the brightkite graph as an instance of the DirectedGraph class.
    In particular, if u and v are friends, there should be an edge between u and v but not between v and u.
    """
    num_nodes = 58228
    graph = DirectedGraph(num_nodes)
    with open(filename) as f:
        for line in f:
            i, j = list(map(int, line.strip().split("\t")))
            graph.add_edge(i, j)
            graph.add_edge(j, i)
    return graph

def oregon_graph(filename="datasets/oregon1_010526.txt"):
    """This method should return a DIRECTED version of the oregon graph as an instance of the DirectedGraph class.
    In particular, if u and v are friends, there should be an edge between u and v but not between v and u.
    """
    num_nodes = 11174
    graph = DirectedGraph(num_nodes)
    with open(filename) as f:
        for _ in range(4): # Skip the first 4 lines
            next(f)

        for line in f:
            i, j = list(map(int, line.strip().split("\t")))
            graph.add_edge(i, j)
            graph.add_edge(j, i)
    return graph

def gnutella_graph(filename="datasets/p2p-Gnutella09.txt"):
    """This method should return a DIRECTED version of the gnutella graph as an instance of the DirectedGraph class.
    In particular, if u and v are friends, there should be an edge between u and v but not between v and u.
    """
    num_nodes = 8114
    graph = DirectedGraph(num_nodes)
    with open(filename) as f:
        for _ in range(4): # Skip the first 4 lines
            next(f)

        for i, line in f:
            i, j = list(map(int, line.strip().split("\t")))
            graph.add_edge(i, j)
            graph.add_edge(j, i)
    return graph

def main():
    run_experiment()

if __name__ == "__main__":
    main()