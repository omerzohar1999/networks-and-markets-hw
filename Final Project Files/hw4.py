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
from scipy.stats import spearmanr
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
            new_weights[i] += (1 - eps) * sum(weights[j] / len(G.edges_from(j)) for j in G.edges_to(i))
            new_weights[i] += 0 if len(G.edges_from(i)) > 0 else (1 - eps) * weights[i] # sink-nodes are treated as self-loops
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
    G.add_edge(0, 4)
    G.add_edge(3, 4)
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
    G = DirectedGraph(10)
    # Shoule form an infinity sign
    # First cycle from nodes 0 to 4
    for i in range(5):
        G.add_edge(i, (i + 1) % 5)
    # Second cycle from nodes 5 to 9
    for i in range(5, 10):
        G.add_edge(i, 5 + (i + 1 - 5) % 5)
    # Connect the two cycles with an edge
    G.add_edge(4, 5)
    return G


def extra_graph_2():
    """This method, should construct and return a DirectedGraph of your choice with at least 10 nodes"""
    G = DirectedGraph(10)
    # Nodes 0-4 are in set A, nodes 5-9 are in set B
    # Create edges from every node in set A to every node in set B
    for i in range(5):
        for j in range(5, 10):
            G.add_edge(i, j)
    return G

def plot_graph(G, filestem="graph", bipartite=False, cycles=None, scores=None):
    import numpy as np
    import random
    n = G.number_of_nodes()
    nodes = list(range(n))
    edges = []

    # Collect all edges in the graph, including self-loops
    for node in nodes:
        for dest in G.edges_from(node):
            edges.append((node, dest))

    positions = {}
    nodes_in_cycles = set()
    node_to_cycle = {}

    if bipartite:
        # Existing bipartite layout code remains unchanged
        mid = n // 2
        left_nodes = nodes[:mid]
        right_nodes = nodes[mid:]

        num_left = len(left_nodes)
        if num_left > 1:
            y_positions_left = np.linspace(0, 10, num_left)
        else:
            y_positions_left = [5]  # Center if only one node

        num_right = len(right_nodes)
        if num_right > 1:
            y_positions_right = np.linspace(0, 10, num_right)
        else:
            y_positions_right = [5]  # Center if only one node

        for idx, node in enumerate(left_nodes):
            positions[node] = np.array([0, y_positions_left[idx]])

        for idx, node in enumerate(right_nodes):
            positions[node] = np.array([10, y_positions_right[idx]])
    else:
        # Initialize positions for cycles
        if cycles is not None:
            cycle_offset = 0
            for cycle in cycles:
                N = len(cycle)
                r = 1.0  # Radius of the circle
                x_center = cycle_offset
                y_center = 0
                if N == 2:
                    # Arrange nodes vertically
                    positions[cycle[0]] = np.array([x_center, y_center + 1])
                    positions[cycle[1]] = np.array([x_center, y_center - 1])
                else:
                    for idx, node in enumerate(cycle):
                        angle = 2 * np.pi * idx / N
                        x = x_center + r * np.cos(angle)
                        y = y_center + r * np.sin(angle)
                        positions[node] = np.array([x, y])
                for node in cycle:
                    nodes_in_cycles.add(node)
                    node_to_cycle[node] = cycle
                cycle_offset += 3  # Shift x_center for next cycle
        else:
            cycles = []

        # Initialize positions for nodes not in cycles
        for i in nodes:
            if i not in positions:
                positions[i] = np.array([random.random(), random.random()])

        # Constants for the force-directed algorithm
        area = 1.0
        k = np.sqrt(area / n)
        iterations = 50
        epsilon = 1e-4
        temperature = area / 10.0

        # Force-directed algorithm to compute node positions
        for _ in range(iterations):
            # Initialize displacements
            displacements = {i: np.array([0.0, 0.0]) for i in nodes}

            # Compute repulsive forces
            for i in nodes:
                for j in nodes:
                    if i != j:
                        delta = positions[i] - positions[j]
                        distance = np.linalg.norm(delta) + epsilon
                        force = (k ** 2) / distance
                        displacements[i] += (delta / distance) * force

            # Compute attractive forces
            for (i, j) in edges:
                delta = positions[i] - positions[j]
                distance = np.linalg.norm(delta) + epsilon
                force = (distance ** 2) / k
                displacements[i] -= (delta / distance) * force
                displacements[j] += (delta / distance) * force

            # Update positions with displacement limits
            for i in nodes:
                if i in nodes_in_cycles:
                    continue  # Skip nodes in cycles
                displacement = displacements[i]
                distance = np.linalg.norm(displacement)
                if distance > 0:
                    displacement = displacement / distance * min(distance, temperature)
                    positions[i] += displacement

            # Cool down temperature
            temperature *= 0.9

        # Normalize positions to fit within a specific range for TikZ
        pos_array = np.array(list(positions.values()))
        min_pos = pos_array.min(axis=0)
        max_pos = pos_array.max(axis=0)
        scale = max(max_pos - min_pos)
        for i in positions:
            positions[i] = (positions[i] - min_pos) / scale * 10  # Scale positions to [0,10]

    # Begin generating TikZ code
    tikz_code = "\\begin{tikzpicture}[node distance=2cm, every loop/.style={min distance=10mm}]\n"

    # Normalize scores if provided
    if scores is not None:
        if isinstance(scores, dict):
            score_values = list(scores.values())
        else:
            score_values = scores
            scores = dict(zip(nodes, scores))

        # Apply a small constant to avoid log(0)
        score_values = np.array(score_values) + 1e-6
        min_score = np.min(score_values)
        max_score = np.max(score_values)

        # Logarithmic scaling
        log_scores = np.log(score_values)  # Apply logarithmic transformation
        min_log_score = np.min(log_scores)
        max_log_score = np.max(log_scores)
        log_score_range = max_log_score - min_log_score + 1e-6  # Avoid division by zero

        # Min-max normalization of log-transformed scores to [0, 1]
        def transform_score(s):
            return (np.log(s + 1e-6) - min_log_score) / log_score_range  # Normalize to [0, 1]

        # Draw nodes with color based on transformed scores
        for i in nodes:
            x, y = positions[i]
            if i in scores:
                normalized_score = transform_score(scores[i])
                # Apply a nonlinear transformation to enhance contrast
                gamma = 0.5  # Adjust this as needed for visibility
                adjusted_score = normalized_score ** gamma
                saturation = 20 + adjusted_score * (80 - 20)  # Adjust saturation range
                fill_color = f"red!{saturation:.0f}"
            else:
                fill_color = "red!50"
            tikz_code += f"\\node[circle, draw=black, fill={fill_color}, inner sep=0pt, minimum size=6mm] (node{i}) at ({x:.2f}, {y:.2f}) {{\\small {i}}};\n"

    else:
        # Draw nodes without scores
        for i in nodes:
            x, y = positions[i]
            fill_color = "red!50"
            tikz_code += f"\\node[circle, draw=black, fill={fill_color}, inner sep=0pt, minimum size=6mm] (node{i}) at ({x:.2f}, {y:.2f}) {{\\small {i}}};\n"

    # Prepare edge sets
    edges_set = set(edges)
    drawn_edges = set()

    # Draw edges
    for (i, j) in edges:
        if (i, j) in drawn_edges:
            continue
        if i == j:
            # Self-loop
            tikz_code += f"\\path[-{'{Stealth[length=3mm, width=2mm]}'}] (node{i}) edge [in=120,out=60,loop] (node{i});\n"
            drawn_edges.add((i, j))
        elif (j, i) in edges_set:
            # Bidirectional edge
            if node_to_cycle.get(i) == node_to_cycle.get(j) and len(node_to_cycle.get(i, [])) == 2:
                # Both nodes are in the same 2-node cycle
                tikz_code += f"\\draw[-{'{Stealth[length=3mm, width=2mm]}'},bend left=30] (node{i}) to (node{j});\n"
                tikz_code += f"\\draw[-{'{Stealth[length=3mm, width=2mm]}'},bend left=30] (node{j}) to (node{i});\n"
            else:
                tikz_code += f"\\draw[-{'{Stealth[length=3mm, width=2mm]}'},bend left] (node{i}) to (node{j});\n"
                tikz_code += f"\\draw[-{'{Stealth[length=3mm, width=2mm]}'},bend left] (node{j}) to (node{i});\n"
            drawn_edges.add((i, j))
            drawn_edges.add((j, i))
        else:
            # Single edge
            tikz_code += f"\\draw[-{'{Stealth[length=3mm, width=2mm]}'}] (node{i}) -- (node{j});\n"
            drawn_edges.add((i, j))

    tikz_code += "\\end{tikzpicture}\n"

    # Write TikZ code to file
    with open(f"figures/{filestem}.tex", 'w') as f:
        f.write(tikz_code)


def q7_analysis():
    """This method runs the scaled_page_rank algorithm on the graphs
    graph_15_1_left, graph_15_1_right, graph_15_2, extra_graph_1, and extra_graph_2
    for 20 iterations, and plots the results with cycle hints."""
    graphs = [
        # (Graph, Filestem, Bipartite, Cycles)
        (graph_15_1_left(), "graph_15_1_left", False, [[0, 1, 2]]),
        (graph_15_1_right(), "graph_15_1_right", False, [[0, 1, 2], [3, 4]]),
        (graph_15_2(), "graph_15_2", False, [[0, 1, 2], [3, 4, 5]]),
        (extra_graph_1(), "extra_graph_1", False, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        (extra_graph_2(), "extra_graph_2", True, None)  # No cycles, bipartite graph
    ]
    for G, name, bipartite, cycles in graphs:
        ranks = scaled_page_rank(G, 20)
        print(f"{name}: {ranks}")
        plot_graph(G, filestem=name, bipartite=bipartite, cycles=cycles, scores=ranks)

# === Problem 8. ===
def facebook_graph(filename="facebook_combined.txt"):
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

def analyze_graph_ranks(graph : DirectedGraph, name="graph"):
    """This method should run the scaled_page_rank algorithm on the given graph for num_iter iterations,
    and return the ranks of the nodes in the graph."""
    latex_name = name.replace("_", r"\_")

    # plot the ranks as one axis and the number of in-links as the other axis
    print(f"Analyzing {name} graph - In-link analysis")
    ranks = scaled_page_rank(graph, 20)
    ordered_ranks_indicies = np.argsort(list(ranks.values()))
    in_links = [len(graph.edges_to(i)) for i in range(graph.number_of_nodes())]
    plt.scatter(x=list(range(len(ordered_ranks_indicies))), y=[in_links[i] for i in ordered_ranks_indicies])
    plt.ylabel("Number of in-links")
    plt.xlabel("PageRank (index in the sorted set of ranks)")
    plt.title(f"PageRank vs. Number of in-links for {latex_name} graph")
    plt.savefig(f"figures/{name}_inlink_analysis.pgf", format="pgf")
    plt.close()

    # calculate spearman correlation
    corr, _ = spearmanr(list(ranks.values()), in_links)
    print(f"Spearman correlation for {name} graph: {corr}")

def q8c_analysis():
    """This method should run the scaled_page_rank algorithm on the facebook graph for 20 iterations,
    and plot the ranks as one axis and the number of in-links as the other axis."""
    graphs = [
        (graph_15_1_left(), "graph_15_1_left"),
        (graph_15_1_right(), "graph_15_1_right"),
        (graph_15_2(), "graph_15_2"),
        (extra_graph_1(), "extra_graph_1"),
        (extra_graph_2(), "extra_graph_2"),
        (facebook_graph(), "facebook")
    ]

    # analysis for each graph
    for graph, name in graphs:
        analyze_graph_ranks(graph, name)

def main():

    ## Problem 7
    # q7_analysis()

    ## Problem 8
    # 8 (a): extract facebook graph
    fb_graph = facebook_graph()

    # 8 (b): run scaled page rank on facebook graph
    ranks = scaled_page_rank(fb_graph, 20)

    # 8 (c): show nodes with highest and lowest ranks
    sorted_ranks = sorted(
        list(range(fb_graph.number_of_nodes())), key=lambda x: ranks[x]
    )
    print("Top 10 nodes:")
    for i in sorted_ranks[-10:]:
        print(i, ranks[i], len(fb_graph.edges_to(i)))

    print("Bottom 10 nodes:")
    for i in sorted_ranks[:10]:
        print(i, ranks[i], len(fb_graph.edges_to(i)))

    q8c_analysis()

if __name__ == "__main__":
    main()
