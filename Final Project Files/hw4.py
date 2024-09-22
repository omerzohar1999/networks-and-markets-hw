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


def plot_graph(G, filestem="graph", bipartite=False):
    n = G.number_of_nodes()
    nodes = list(range(n))
    edges = []

    # Collect all edges in the graph, including self-loops
    for node in nodes:
        for dest in G.edges_from(node):
            edges.append((node, dest))

    positions = {}

    if bipartite:
        # Divide nodes into two groups
        mid = n // 2
        left_nodes = nodes[:mid]
        right_nodes = nodes[mid:]

        # Determine vertical positions for left group
        num_left = len(left_nodes)
        if num_left > 1:
            y_positions_left = np.linspace(0, 10, num_left)
        else:
            y_positions_left = [5]  # Center if only one node

        # Determine vertical positions for right group
        num_right = len(right_nodes)
        if num_right > 1:
            y_positions_right = np.linspace(0, 10, num_right)
        else:
            y_positions_right = [5]  # Center if only one node

        # Assign positions to left group nodes
        for idx, node in enumerate(left_nodes):
            positions[node] = np.array([0, y_positions_left[idx]])

        # Assign positions to right group nodes
        for idx, node in enumerate(right_nodes):
            positions[node] = np.array([10, y_positions_right[idx]])
    else:
        # Initialize positions randomly within a unit square
        positions = {i: np.array([random.random(), random.random()]) for i in nodes}

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
    tikz_code = "\\begin{tikzpicture}[>=stealth, node distance=2cm, every loop/.style={min distance=10mm}]\n"

    # Draw nodes
    for i in nodes:
        x, y = positions[i]
        tikz_code += f"\\node[circle, draw=black, fill=red!50, inner sep=0pt, minimum size=6mm] (node{i}) at ({x:.2f}, {y:.2f}) {{\\small {i}}};\n"

    # Draw edges
    for (i, j) in edges:
        if i == j:
            # Self-loop
            tikz_code += f"\\path[->] (node{i}) edge [in=120,out=60,loop] (node{i});\n"
        else:
            # Adjust edge style for bipartite graphs
            if bipartite:
                # Use bend left/right for better visualization
                tikz_code += f"\\draw[->] (node{i}) -- (node{j});\n"
            else:
                tikz_code += f"\\draw[->] (node{i}) -- (node{j});\n"

    tikz_code += "\\end{tikzpicture}\n"

    # Write TikZ code to file
    with open(f"figures/{filestem}.tex", 'w') as f:
        f.write(tikz_code)

def q7_analysis():
    """This method should run the scaled_page_rank algorithm on the graphs
    graph_15_1_left, graph_15_1_right, graph_15_2, extra_graph_1, and extra_graph_2
    for 20 iterations, and print the results."""
    for graph_content in [(graph_15_1_left(), "graph_15_1_left"), (graph_15_1_right(), "graph_15_1_right"), (graph_15_2(), "graph_15_2"), (extra_graph_1(), "extra_graph_1"), (extra_graph_2(), "extra_graph_2", True)]:
        print(scaled_page_rank(graph_content[0], 20))
        plot_graph(*graph_content)

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

    ## Problem 7
    # q7_analysis()

    ## Problem 8
    fb_graph = facebook_graph()
    ranks = scaled_page_rank(fb_graph, 25)
    # 8 (c): show nodes with highest and lowest ranks
    sorted_ranks = sorted(
        list(range(fb_graph.number_of_nodes())), key=lambda x: ranks[x]
    )
    print("Top 10 nodes:")
    for i in sorted_ranks[-10:]:
        print(i, ranks[i], len(fb_graph.edges_from(i)))
    print("Bottom 10 nodes:")
    for i in sorted_ranks[:10]:
        print(i, ranks[i], len(fb_graph.edges_from(i)))


if __name__ == "__main__":
    main()
