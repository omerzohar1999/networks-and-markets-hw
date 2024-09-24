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
        self.edges = dict()          # Outgoing edges: node -> set of destination nodes
        self.reverse_edges = dict()  # Incoming edges: node -> set of origin nodes
        self.degrees = [0] * self.n  # Degree of each node (for undirected graph, degree = in-degree + out-degree)

    def add_edge(self, origin_node, destination_node):
        """Adds an edge from origin_node to destination_node."""
        # Add edge to outgoing edges
        if origin_node not in self.edges:
            self.edges[origin_node] = set()
        self.edges[origin_node].add(destination_node)

        # Add edge to incoming edges
        if destination_node not in self.reverse_edges:
            self.reverse_edges[destination_node] = set()
        self.reverse_edges[destination_node].add(origin_node)

        # Update degrees (since the graph is undirected in your use case)
        self.degrees[origin_node] += 1
        self.degrees[destination_node] += 1

    def edges_from(self, origin_node):
        """Returns a list of all destination nodes such that there is an edge (origin_node, destination_node)."""
        return list(self.edges.get(origin_node, set()))

    def edges_to(self, destination_node):
        """Returns a list of all origin nodes such that there is an edge (origin_node, destination_node)."""
        return list(self.reverse_edges.get(destination_node, set()))

    def get_edge(self, origin_node, destination_node):
        """Returns True if there is an edge from origin_node to destination_node, False otherwise."""
        return destination_node in self.edges.get(origin_node, set())

    def number_of_nodes(self):
        """Returns the number of nodes in the graph."""
        return self.n

    def get_degrees(self):
        """Returns a list of degrees for all nodes."""
        return self.degrees

    def to_adjacency_matrix(self):
        """Converts the graph to an adjacency matrix."""
        adj_matrix = np.zeros((self.n, self.n))
        for i in self.edges:
            for j in self.edges[i]:
                adj_matrix[i, j] = 1
        return adj_matrix

    def to_adjacency_matrix_subgraph(self, nodes):
        """Creates an adjacency matrix for a subgraph induced by the given nodes."""
        idx_map = {node: idx for idx, node in enumerate(nodes)}
        size = len(nodes)
        adj_matrix = np.zeros((size, size))
        for i in nodes:
            idx_i = idx_map[i]
            neighbors = self.edges_from(i)
            for j in neighbors:
                if j in idx_map:
                    idx_j = idx_map[j]
                    adj_matrix[idx_i, idx_j] = 1
        return adj_matrix, idx_map

    def floyd_warshall(self):
        """Computes the reachability matrix using the Floyd-Warshall algorithm from scipy."""
        # Convert graph to adjacency matrix
        adj_matrix = self.to_adjacency_matrix()

        # Convert adjacency matrix to compressed sparse row (CSR) format
        graph = csr_matrix(adj_matrix)

        # Run Floyd-Warshall algorithm
        dist_matrix = floyd_warshall(csgraph=graph, directed=True, unweighted=True)

        # Return the distance matrix
        return dist_matrix

    def reverse_reachability_weighted(self):
        """Returns a dictionary where keys are nodes and values are the weighted reverse reachability."""
        dist = self.floyd_warshall()
        rr = np.zeros(self.n)
        for i in range(self.n):
            rr[i] = sum(dist[j, i] for j in range(self.n) if dist[j, i] != np.inf)
        return {i: rr[i] for i in range(self.n)}

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
    active = set(initial_infected)
    healthy = set(range(G.number_of_nodes())) - infected - vaccinated

    while active:
        new_active = set()
        for u in active:
            for v in G.edges_from(u):
                if v in healthy and random.random() <= p:
                    healthy.remove(v)
                    infected.add(v)
                    new_active.add(v)
        active = new_active
    return infected

def simulate_SIR(G: DirectedGraph, p: float, delta: float, initial_infected: set, vaccinated: set):
    """Simulate the SIR model once.

    Args:
        G (DirectedGraph): The graph.
        p (float): The infection probability.
        delta (float): The recovery probability.
        initial_infected (set): The set of initially infected nodes.
        vaccinated (set): The set of vaccinated nodes.

    Returns:
        tuple: The set of infected and recovered nodes at the end of the simulation.
    """
    susceptible = set(range(G.number_of_nodes())) - initial_infected - vaccinated
    infected = set(initial_infected)
    recovered = set()
    
    while infected:
        new_infected = set()
        still_infected = set()
        for u in infected:
            # Infection step
            for v in G.edges_from(u):
                if v in susceptible and random.random() <= p:
                    susceptible.remove(v)
                    new_infected.add(v)
            # Recovery step
            if random.random() <= delta:
                recovered.add(u)
            else:
                still_infected.add(u)
        # Update infected set
        infected = still_infected.union(new_infected)
    return recovered

def average_infected(G: DirectedGraph, p: float, initial_infected: set, vaccinated: set, num_simulations: int = 100, model='IC', delta: float = 0.1):
    """Compute the average number of infected nodes over multiple simulations.

    Args:
        G (DirectedGraph): The graph.
        p (float): The propagation probability.
        initial_infected (set): The set of initially infected nodes.
        vaccinated (set): The set of vaccinated nodes.
        num_simulations (int): Number of simulations to run.
        model (str): 'IC' or 'SIR'
        delta (float): Recovery probability for SIR model.

    Returns:
        float: The average number of infected nodes.
    """
    total_infected = 0
    for _ in range(num_simulations):
        if model == 'IC':
            infected = simulate_IC(G, p, initial_infected, vaccinated)
            total_infected += len(infected)
        elif model == 'SIR':
            recovered = simulate_SIR(G, p, delta, initial_infected, vaccinated)
            total_infected += len(recovered)
        else:
            raise ValueError(f"Unknown model: {model}")
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

def pagerank_strategy(G: DirectedGraph, k: int, healthy_nodes: set, num_iter: int = 10, ranks = None):
    """Select k healthy nodes with the highest PageRank scores.

    Args:
        G (DirectedGraph): The graph.
        k (int): Number of nodes to select.
        healthy_nodes (set): Set of healthy nodes.
        num_iter (int): Number of iterations for PageRank.

    Returns:
        set: Set of selected nodes.
    """
    pr = scaled_page_rank(G, num_iter) if ranks is None else ranks
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
    import numpy as np

    # Map healthy nodes to indices from 0 to h-1
    healthy_nodes_list = list(healthy_nodes)
    node_to_index = {node: idx for idx, node in enumerate(healthy_nodes_list)}
    index_to_node = {idx: node for idx, node in enumerate(healthy_nodes_list)}
    h = len(healthy_nodes_list)

    # Create adjacency matrix A_sub of size h x h
    # A_sub corresponds to the adjacency matrix A in the algorithm
    A_sub = np.zeros((h, h))
    for i in healthy_nodes_list:
        idx_i = node_to_index[i]
        neighbors_i = set(G.edges_from(i)) | set(G.edges_to(i))
        for j in neighbors_i:
            if j in healthy_nodes:
                idx_j = node_to_index[j]
                A_sub[idx_i, idx_j] = 1

    # Make adjacency matrix symmetric to treat the graph as undirected
    A_sub = np.maximum(A_sub, A_sub.T)

    # Step 1: Compute the leading eigenvalue and eigenvector
    eigenvalues, eigenvectors = np.linalg.eigh(A_sub)
    lambda1 = eigenvalues[-1]
    u = eigenvectors[:, -1]

    # Step 2: Initialize the selected set S
    S = []
    S_indices = []

    # Precompute v(j) = 2 * lambda1 * u(j)^2 (since A(j,j) = 0)
    v = 2 * lambda1 * u**2

    # Initialize b as zeros (will be updated iteratively)
    b = np.zeros(h)

    # Step 4: Iteratively select nodes
    for _ in range(k):
        max_score = -np.inf
        max_node = None
        for i in range(h):
            if i in S_indices:
                continue
            # b(i) = sum over s in S of A(i,s) * u(s)
            # Since b is A(:, S) * u(S), we can compute b(i) incrementally
            # In the first iteration, b is zero since S is empty
            # We use b[i] directly since it accumulates over iterations
            score = v[i] - 2 * u[i] * b[i]
            if score > max_score:
                max_score = score
                max_node = i
        if max_node is not None:
            S.append(index_to_node[max_node])
            S_indices.append(max_node)
            # Update b for all nodes
            # b = A(:, S) * u(S)
            b += A_sub[:, max_node] * u[max_node]
        else:
            break  # No more nodes to select

    selected = set(S)
    return selected

def run_experiment():
    """Run the experiments on multiple graphs and plot the results."""

    # Graph loaders and parameters
    graph_loaders = {
        'Facebook': 
            {
                'create': facebook_graph,
                'k_values': [5, 10, 20, 30, 40, 50],
                'num_simulations': 50,
                'num_initial_infected': 100
            },
        # 'Brightkite':
        #     {
        #         'create': brightkite_graph,
        #         'k': [10, 30, 50, 70, 90, 110, 130, 150],
        #         'num_simulations': 40,
        #         'initial_infected': 400
        #     },
        # 'Oregon':
        #     {
        #         'create': oregon_graph,
        #         'k': [5, 10, 20, 30, 40, 50],
        #         'num_simulations': 50,
        #         'initial_infected': 100
        #     },
        # 'Gnutella':
        #     {
        #         'create': gnutella_graph,
        #         'k': [10, 30, 50, 70, 90, 110, 130, 150],
        #         'num_simulations': 50,
        #         'initial_infected': 100
        #     }
    }

    # Vaccination strategies
    strategies = {
        'Random': random_strategy,
        'Degree': degree_strategy,
        'PageRank': pagerank_strategy,
        'NetShield': netshield
    }

    # Propagation configurations
    propagation_configs = [
        {'p': 0.1, 'model': 'IC', 'delta': None},
        # {'p': 0.5, 'model': 'SIR', 'delta': 0.1}
    ]

    # Run experiments
    for graph_name, graph_params in graph_loaders.items():

        # Load the graph
        G = graph_params['create']()

        # Run experiments for each propagation configuration
        for config in propagation_configs:

            # Load propagation parameters
            p = config['p']
            model = config['model']
            delta = config['delta']
            print(f"\nRunning experiments on {graph_name} graph with p={p}, model={model}")

            # Run experiments for each strategy
            results = {strategy: [] for strategy in strategies}

            for name, strategy in strategies.items():
                
                # If this is the scaled PageRank strategy, precompute the ranks
                ranks : dict = None
                if name == 'PageRank':
                    ranks = scaled_page_rank(G, 10)

                # For each vaccination budget for the graph
                for k in graph_params['k_values']:
                    
                    print(f"\nVaccination budget (k): {k}")

                    # Randomly choose initial infected nodes
                    initial_infected = set(random.sample(range(G.number_of_nodes()), graph_params['num_initial_infected']))
                    healthy_nodes = set(range(G.number_of_nodes())) - initial_infected

                    # Select vaccinated nodes
                    if name == 'Random':
                        vaccinated = strategy(k, healthy_nodes)
                    elif name == 'PageRank':
                        vaccinated = strategy(G, k, healthy_nodes, ranks=ranks)
                    else:
                        vaccinated = strategy(G, k, healthy_nodes)

                    # Simulate the spread
                    avg_infected = average_infected(G, p, initial_infected, vaccinated, num_simulations=graph_params['num_simulations'], model=model, delta=delta)
                    avg_healthy = G.number_of_nodes() - avg_infected
                    results[name].append(avg_healthy)
                    print(f"Strategy: {name}, Avg Healthy Nodes: {avg_healthy}")

            # Plotting the results
            plt.figure()
            for name in strategies:
                plt.plot(graph_params['k_values'], results[name], label=name)
            plt.xlabel('Number of Vaccines (k)')
            plt.ylabel('Average Number of Healthy Nodes')
            plt.title(f'Effectiveness of Vaccination Strategies ({graph_name} Graph, p={p}, model={model})')
            plt.legend()
            plt.savefig(f'figures/vaccination_{graph_name}_p{p}_model{model}.png', format='png')
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