# Do not include any other files or an external package, unless it is one of
# [numpy, pandas, scipy, matplotlib, random]
# please contact us before sumission if you want another package approved.
import random
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Tuple, Callable, Union


class DirectedGraph:
    """
    This class represents a directed graph. The graph is stored as a dictionary of sets, where each key is a node and the corresponding value is a set of nodes that are reachable from the key node. The graph is assumed to be undirected, so if there is an edge from node i to node j, there is also an edge from node j to node i.
    """
    def __init__(self, number_of_nodes: int) -> None:
        """Initializes a DirectedGraph object with the specified number of nodes.

        Args:
            number_of_nodes (int): The number of nodes in the graph.

        Assume that nodes are represented by indices/integers between 0 and number_of_nodes - 1.
        """
        self.n = number_of_nodes
        self.edges = dict()          # Outgoing edges: node -> set of destination nodes
        self.reverse_edges = dict()  # Incoming edges: node -> set of origin nodes
        self.degrees = [0] * self.n  # Degree of each node (for undirected graph, degree = in-degree + out-degree)

    def add_edge(self, origin_node: int, destination_node: int) -> None:
        """Adds an edge from origin_node to destination_node.

        Args:
            origin_node (int): The starting node of the edge.
            destination_node (int): The ending node of the edge.
        """
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

    def edges_from(self, origin_node: int) -> List[int]:
        """Returns a list of all destination nodes that are reachable from the origin_node.

        Args:
            origin_node (int): The starting node.

        Returns:
            List[int]: A list of destination nodes.
        """
        return list(self.edges.get(origin_node, set()))

    def edges_to(self, destination_node: int) -> List[int]:
        """Returns a list of all origin nodes that have edges leading to the destination_node.

        Args:
            destination_node (int): The destination node.

        Returns:
            List[int]: A list of origin nodes.
        """
        return list(self.reverse_edges.get(destination_node, set()))

    def get_edge(self, origin_node: int, destination_node: int) -> bool:
        """Checks whether there is an edge from origin_node to destination_node.

        Args:
            origin_node (int): The starting node.
            destination_node (int): The ending node.

        Returns:
            bool: True if the edge exists, False otherwise.
        """
        return destination_node in self.edges.get(origin_node, set())

    def number_of_nodes(self) -> int:
        """Returns the total number of nodes in the graph.

        Returns:
            int: The number of nodes.
        """
        return self.n

    def get_degrees(self) -> List[int]:
        """Returns the degree of each node in the graph.

        Returns:
            List[int]: A list of degrees for all nodes.
        """
        return self.degrees

    def to_adjacency_matrix(self) -> np.ndarray:
        """Converts the directed graph to an adjacency matrix.

        Returns:
            np.ndarray: The adjacency matrix representation of the graph.
        """
        adj_matrix = np.zeros((self.n, self.n))
        for i in self.edges:
            for j in self.edges[i]:
                adj_matrix[i, j] = 1
        return adj_matrix

    def to_adjacency_matrix_subgraph(self, nodes: List[int]) -> Tuple[np.ndarray, Dict[int, int]]:
        """Creates an adjacency matrix for a subgraph induced by the given set of nodes.

        Args:
            nodes (List[int]): A list of nodes to induce the subgraph.

        Returns:
            Tuple[np.ndarray, Dict[int, int]]: The adjacency matrix of the subgraph and a mapping of nodes to matrix indices.
        """
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

    def floyd_warshall(self) -> np.ndarray:
        """Computes the reachability matrix using the Floyd-Warshall algorithm.

        Returns:
            np.ndarray: The distance matrix representing shortest paths between nodes.
        """
        # Convert graph to adjacency matrix
        adj_matrix = self.to_adjacency_matrix()

        # Convert adjacency matrix to compressed sparse row (CSR) format
        graph = csr_matrix(adj_matrix)

        # Run Floyd-Warshall algorithm
        dist_matrix = floyd_warshall(csgraph=graph, directed=True, unweighted=True)

        # Return the distance matrix
        return dist_matrix

    def reverse_reachability_weighted(self) -> Dict[int, float]:
        """Computes the weighted reverse reachability for each node.

        Returns:
            Dict[int, float]: A dictionary where keys are node indices and values are their weighted reverse reachability scores.
        """
        dist = self.floyd_warshall()
        rr = np.zeros(self.n)
        for i in range(self.n):
            rr[i] = sum(dist[j, i] for j in range(self.n) if dist[j, i] != np.inf)
        return {i: rr[i] for i in range(self.n)}

def scaled_page_rank(
    G: DirectedGraph,
    num_iter: int,
    eps: float = 1 / 7.0
) -> Dict[int, float]:
    """Runs the epsilon-scaled PageRank algorithm on a directed graph.

    Args:
        G (DirectedGraph): The directed graph.
        num_iter (int): The number of iterations to run.
        eps (float): The epsilon scaling factor for PageRank, default is 1/7.0.

    Returns:
        Dict[int, float]: A dictionary where each key is a node and the value is its PageRank score.

    This method, given a DirectedGraph G, runs the epsilon-scaled page-rank algorithm for num-iter iterations, for parameter eps, and returns a Dictionary where the keys are the set of nodes [0,...,G.number_of_nodes() - 1], each associated with a value equal to the score of output by the eps-scaled pagerank algorithm.

    In the case of num_iter=0, all nodes should have weight 1/G.number_of_nodes()
    """
    weights = [1 / G.number_of_nodes()] * G.number_of_nodes()
    for _ in range(num_iter):
        new_weights = [eps / G.number_of_nodes()] * G.number_of_nodes()
        for i in range(G.number_of_nodes()):
            new_weights[i] += (1 - eps) * sum(weights[j] / len(G.edges_from(j)) for j in G.edges_to(i))
            new_weights[i] += 0 if len(G.edges_from(i)) > 0 else (1 - eps) * weights[i] # sink-nodes are treated as self-loops
        weights = new_weights

    return {i: weights[i] for i in range(G.number_of_nodes())}

def simulate_IC(
    G: DirectedGraph,
    p: float,
    initial_infected: set,
    vaccinated: set
) -> set:
    """Simulates the Independent Cascade (IC) model on a directed graph.

    Args:
        G (DirectedGraph): The graph.
        p (float): The propagation probability.
        initial_infected (set): A set of initially infected nodes.
        vaccinated (set): A set of vaccinated nodes.

    Returns:
        set: A set of infected nodes after the simulation.
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

def simulate_SIR(
    G: DirectedGraph,
    p: float,
    delta: float,
    initial_infected: set,
    vaccinated: set
) -> set:
    """Simulates the Susceptible-Infected-Recovered (SIR) model on a directed graph.

    Args:
        G (DirectedGraph): The graph.
        p (float): The infection probability.
        delta (float): The recovery probability.
        initial_infected (set): A set of initially infected nodes.
        vaccinated (set): A set of vaccinated nodes.

    Returns:
        Tuple[set, set]: A tuple containing the set of infected nodes and the set of recovered nodes.
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

def average_infected(
    G: DirectedGraph,
    p: float,
    initial_infected: set,
    vaccinated: set,
    num_simulations: int = 100,
    model='IC',
    delta: float = 0.1
) -> float:
    """Computes the average number of infected nodes across multiple simulations.

    Args:
        G (DirectedGraph): The graph.
        p (float): The propagation probability.
        initial_infected (set): A set of initially infected nodes.
        vaccinated (set): A set of vaccinated nodes.
        num_simulations (int): The number of simulations to run. Default is 100.
        model (str): The infection model ('IC' or 'SIR'). Default is 'IC'.
        delta (float): The recovery probability for the SIR model. Default is 0.1.

    Returns:
        float: The average number of infected nodes over the simulations.
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

def degree_strategy(
    G: DirectedGraph,
    k: int,
    healthy_nodes: set
) -> set:
    """Selects k healthy nodes with the highest degrees in the graph.

    Args:
        G (DirectedGraph): The graph.
        k (int): The number of nodes to select.
        healthy_nodes (set): A set of healthy nodes.

    Returns:
        set: A set of selected nodes with the highest degrees.
    """
    degrees = G.get_degrees()
    degree_list = [(i, degrees[i]) for i in healthy_nodes]
    degree_list.sort(key=lambda x: x[1], reverse=True)
    selected = {node for node, degree in degree_list[:k]}
    return selected

def pagerank_strategy(
    G: DirectedGraph,
    k: int,
    healthy_nodes: set,
    num_iter: int = 10,
    ranks = None
) -> set:
    """Selects k healthy nodes based on their PageRank scores.

    Args:
        G (DirectedGraph): The graph.
        k (int): The number of nodes to select.
        healthy_nodes (set): A set of healthy nodes.
        num_iter (int): The number of iterations for PageRank. Default is 10.
        ranks (Dict[int, float], optional): Precomputed PageRank scores. If None, PageRank will be computed.

    Returns:
        set: A set of selected nodes based on PageRank scores.
    """
    pr = scaled_page_rank(G, num_iter) if ranks is None else ranks
    pr_list = [(i, pr[i]) for i in healthy_nodes]
    pr_list.sort(key=lambda x: x[1], reverse=True)
    selected = {node for node, score in pr_list[:k]}
    return selected

def random_strategy(k: int, healthy_nodes: Set[int]) -> Set[int]:
    """Randomly selects k healthy nodes from the set of healthy nodes.

    Args:
        k (int): The number of nodes to select.
        healthy_nodes (Set[int]): A set of healthy nodes.

    Returns:
        Set[int]: A set of randomly selected healthy nodes.
    """
    selected = set(random.sample(list(set(healthy_nodes)), k))
    return selected

def netshield(
    G: DirectedGraph,
    k: int,
    healthy_nodes: Set[int]
) -> Set[int]:
    """Implements the NETSHIELD algorithm to select k nodes for vaccination.

    Args:
        G (DirectedGraph): The graph.
        k (int): The number of nodes to select.
        healthy_nodes (set): A set of healthy nodes.

    Returns:
        set: A set of selected nodes using the NETSHIELD algorithm.
    """
    # Map healthy nodes to indices from 0 to h-1
    healthy_nodes_list = list(healthy_nodes)
    node_to_index = {node: idx for idx, node in enumerate(healthy_nodes_list)}
    index_to_node = {idx: node for node, idx in node_to_index.items()}
    h = len(healthy_nodes_list)

    # Build sparse adjacency matrix A_sub in COO format
    row: List[int] = []
    col: List[int] = []

    for i in healthy_nodes_list:
        idx_i = node_to_index[i]
        neighbors_i = set(G.edges_from(i)) | set(G.edges_to(i))
        for j in neighbors_i:
            if j in healthy_nodes:
                idx_j = node_to_index[j]
                row.append(idx_i)
                col.append(idx_j)

    data = np.ones(len(row))
    A_sub = coo_matrix((data, (row, col)), shape=(h, h)).tocsr()

    # Make adjacency matrix symmetric to treat the graph as undirected
    A_sub = A_sub.maximum(A_sub.transpose())

    # Step 1: Compute the leading eigenvalue and eigenvector using sparse methods
    lambda1, u = eigsh(A_sub, k=1, which='LA')  # LA: Largest Algebraic eigenvalue
    lambda1 = lambda1[0]
    u = u[:, 0]

    # Step 2: Initialize the selected set S
    S: List[int] = []
    S_indices: Set[int] = set()

    # Precompute v(j) = 2 * lambda1 * u(j)^2
    v = 2 * lambda1 * u**2

    # Initialize b as zeros (will be updated iteratively)
    b = np.zeros(h)

    # For efficient access
    u = u.reshape(-1, 1)

    # Step 4: Iteratively select nodes
    for _ in range(k):
        # Create a mask for unselected nodes
        mask = np.ones(h, dtype=bool)
        if S_indices:
            mask[list(S_indices)] = False

        # Compute score only for unselected nodes
        score = np.full(h, -np.inf)
        unselected_indices = np.where(mask)[0]
        score_unselected = v[unselected_indices] - 2 * u[unselected_indices].flatten() * b[unselected_indices]
        score[unselected_indices] = score_unselected

        # Select node with maximum score
        max_idx = np.argmax(score)
        max_node = index_to_node[max_idx]
        S.append(max_node)
        S_indices.add(max_idx)

        # Update b: b = b + A_sub[:, max_idx] * u[max_idx]
        b += A_sub[:, max_idx].toarray().flatten() * u[max_idx, 0]

    selected = set(S)
    return selected

def run_experiment() -> None:
    """Runs the experiments on multiple graphs using different vaccination strategies and plots the results."""

    # Graph loaders and parameters
    graph_loaders: Dict[str, Dict[str, Union[Callable[[], DirectedGraph], List[int], int]]] = {
        'Facebook': 
            {
                'create': facebook_graph,
                'k_values': [10, 30, 50, 70, 90, 110, 130, 150],
                'num_simulations': 200,
                'num_initial_infected': 100
            },
        'Brightkite':
            {
                'create': brightkite_graph,
                'k_values': [10, 50, 100, 150, 200, 250, 300, 350],
                'num_simulations': 50,
                'num_initial_infected': 400
            },
        'LastFM':
            {
                'create': lastfm_graph,
                'k_values': [10, 30, 50, 70, 90, 110, 130, 150],
                'num_simulations': 200,
                'num_initial_infected': 100
            },
        'Gnutella':
            {
                'create': gnutella_graph,
                'k_values': [10, 30, 50, 70, 90, 110, 130, 150],
                'num_simulations': 200,
                'num_initial_infected': 100
            }
    }

    # Vaccination strategies
    strategies: Dict[str, Callable[[DirectedGraph, int, Set[int]], Set[int]]] = {
        'NoVax': lambda x, y, z: set(),
        'Random': random_strategy,
        'Degree': degree_strategy,
        'PageRank': pagerank_strategy,
        'NetShield': netshield
    }

    # Define line styles, markers, and colors for each strategy
    line_styles: Dict[str, Tuple[str, str, str]] = {
        'NoVax': ('-', 'blue', 'o'),      # Solid line, blue, circle marker
        'Random': ('--', 'green', 's'),   # Dashed line, green, square marker
        'Degree': (':', 'red', '^'),      # Dotted line, red, triangle-up marker
        'PageRank': ('-.', 'purple', 'D'), # Dash-dot line, purple, diamond marker
        'NetShield': ('-', 'orange', 'x') # Solid line, orange, cross marker
    }

    # Propagation configurations
    propagation_configs: List[Dict[str, Union[float, str, float]]] = [
        {'p': 0.4, 'model': 'IC', 'delta': None},
        {'p': 0.8, 'model': 'SIR', 'delta': 0.1}
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
            results: Dict[str, List[float]] = {strategy: [] for strategy in strategies}

            for name, strategy in strategies.items():
                
                # If this is the scaled PageRank strategy, precompute the ranks
                ranks: Dict[int, float] = None
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
            plt.figure(figsize=(8, 6))
            for name in strategies:
                line_style, color, marker = line_styles[name]  # Get the line style, color, and marker
                plt.plot(graph_params['k_values'], results[name], label=name, linestyle=line_style, color=color, marker=marker, markersize=8)
            plt.xlabel('Number of Vaccines (k)')
            plt.ylabel('Average Number of Healthy Nodes')
            plt.title(f'Effectiveness of Vaccination Strategies ({graph_name} Graph, p={p}, model={model})')
            plt.legend()
            plt.savefig(f'figures/vaccination_{graph_name}_p{p}_model{model}.png', format='png')
            plt.savefig(f'figures/vaccination_{graph_name}_p{p}_model{model}.pgf', format='pgf')

def facebook_graph(filename: str = "datasets/facebook_combined.txt") -> DirectedGraph:
    """Loads the Facebook social network graph and returns it as a DirectedGraph.

    Args:
        filename (str): The file path of the Facebook graph dataset.

    Returns:
        DirectedGraph: The Facebook social network graph.
    
    This method should return a DIRECTED version of the facebook graph as an instance of the DirectedGraph class. In particular, if u and v are friends, there should be an edge between u and v and an edge between v and u.

    We assume no header in the file. Each line contains two integers representing an edge in the graph.
    """
    num_nodes = 4039
    graph = DirectedGraph(num_nodes)
    with open(filename) as f:
        for line in f:
            i, j = list(map(int, line.strip().split(" ")))
            graph.add_edge(i, j)
            graph.add_edge(j, i)
    return graph

def brightkite_graph(filename: str = "datasets/brightkite.txt") -> DirectedGraph:
    """Loads the Brightkite social network graph and returns it as a DirectedGraph.

    Args:
        filename (str): The file path of the Brightkite graph dataset.

    Returns:
        DirectedGraph: The Brightkite social network graph.
        
    This method should return a DIRECTED version of the brightkite graph as an instance of the DirectedGraph class. In particular, if u and v are friends, there should be an edge between u and v but not between v and u.

    We assume no header in the file. Each line contains two integers representing an edge in the graph.
    """
    num_nodes = 58228
    graph = DirectedGraph(num_nodes)
    with open(filename) as f:
        for line in f:
            i, j = list(map(int, line.strip().split("\t")))
            graph.add_edge(i, j)
            graph.add_edge(j, i)
    return graph

def lastfm_graph(filename: str = "datasets/lastfm_asia_edges.csv") -> DirectedGraph:
    """Loads the LastFM social network graph and returns it as a DirectedGraph.

    Args:
        filename (str): The file path of the LastFM graph dataset.

    Returns:
        DirectedGraph: The LastFM social network graph.

    This method should return a DIRECTED version of the lastfm graph as an instance of the DirectedGraph class. In particular, if u and v are friends, there should be an edge between u and v but not between v and u.

    We assume the first line of the file is a header and should be skipped. The remaining lines contain pairs of integers representing edges in the graph.
    """
    num_nodes = 7624
    graph = DirectedGraph(num_nodes)
    with open(filename) as f:
        for _ in range(1): # Skip the first 4 lines
            next(f)

        for line in f:
            i, j = list(map(int, line.strip().split(",")))
            graph.add_edge(i, j)
            graph.add_edge(j, i)
    return graph

def gnutella_graph(filename: str = "datasets/p2p-Gnutella08.txt") -> DirectedGraph:
    """Loads the Gnutella peer-to-peer network graph and returns it as a DirectedGraph.

    Args:
        filename (str): The file path of the Gnutella graph dataset.

    Returns:
        DirectedGraph: The Gnutella peer-to-peer network graph.

    This method should return a DIRECTED version of the gnutella graph as an instance of the DirectedGraph class. In particular, if u and v are friends, there should be an edge between u and v but not between v and u.

    We assume the first 4 lines of the file are metadata and should be skipped. The remaining lines contain pairs of integers representing edges in the graph.
    """
    num_nodes = 8114
    graph = DirectedGraph(num_nodes)
    with open(filename) as f:
        for _ in range(4): # Skip the first 4 lines
            next(f)

        for line in f:
            i, j = list(map(int, line.strip().split("\t")))
            graph.add_edge(i, j)
            graph.add_edge(j, i)
    return graph

def main():
    run_experiment()

if __name__ == "__main__":
    main()