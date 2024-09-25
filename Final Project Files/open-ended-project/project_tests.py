import random
import unittest
from itertools import product
from project import *

####################################################################################################
########### REFERENCE IMPLEMENTATION ##############################################################
####################################################################################################

# Create a namespace 'reference'
class reference:
    # Implement the methods in this class as appropriate. Feel free to add other methods
    # and attributes as needed. 
    # Assume that nodes are represented by indices between 0 and number_of_nodes - 1
    class DirectedGraph:
        def __init__(self, number_of_nodes):
            self.__number_of_nodes = number_of_nodes
            self.adjacency_matrix = np.zeros((number_of_nodes, number_of_nodes))

        def add_edge(self, origin_node, destination_node):
            self.adjacency_matrix[origin_node, destination_node] = 1
        
        def edges_from(self, origin_node):
            '''This method should return a list of all the nodes u such that the edge (origin_node,u) is
            part of the graph.'''
            get_row = self.adjacency_matrix[origin_node]
            get_edge_nodes = np.where(get_row == 1)[0]
            return list(get_edge_nodes)
        
        def check_edge(self, origin_node, destination_node):
            '''This method should return true if there is an edge between origin_node and destination_node,
            and false otherwise'''
            return self.adjacency_matrix[origin_node, destination_node] == 1

        def number_of_nodes(self):
            '''This method should return the number of nodes in the graph'''
            return self.__number_of_nodes

        # Additional Methods
        def out_degree(self, origin_node):
            return len(self.edges_from(origin_node=origin_node))
        
        def get_nodes(self):
            return list(range(self.number_of_nodes()))
        
        def edges_to(self, destination_node):
            return list(np.where(self.adjacency_matrix[:, destination_node] == 1)[0])
    
    @staticmethod
    def scaled_page_rank(graph, num_iter, eps=1/7.0, enforce_sink=False):
        '''This method, given a directed graph, should run the epsilon-scaled page-rank
        algorithm for num_iter iterations and return a mapping (dictionary) between a node and its weight. 
        In the case of 0 iterations, all nodes should have weight 1/number_of_nodes'''  
        previous = np.full(graph.number_of_nodes(), (1 / graph.number_of_nodes()))
        ranks = np.zeros(graph.number_of_nodes())

        for i in range(num_iter):
            for j in range(graph.number_of_nodes()):
                rank = 0
                for node in graph.edges_to(j):
                    rank += previous[node] / graph.out_degree(node)
                rank += 0 if (graph.out_degree(j) > 0 or (not enforce_sink)) else previous[j] # If the node is a sink, add the previous rank
                ranks[j] = (eps / graph.number_of_nodes()) + (1 - eps) * rank
            previous = np.copy(ranks)

        # Return as a dictionary mapping nodes to their scores
        return {node: score for node, score in enumerate(previous)}
    
    @staticmethod
    def graph_15_1_left():
        '''This method should construct and return a DirectedGraph encoding the left example in fig 15.1
        Use the following indexes: A:0, B:1, C:2, Z:3'''
        g = reference.DirectedGraph(4)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        g.add_edge(0, 3)
        g.add_edge(3, 3)
        return g
    
    @staticmethod
    def graph_15_1_right():
        '''This method should construct and return a DirectedGraph encoding the right example in fig 15.1
        Use the following indexes: A:0, B:1, C:2, Z1:3, Z2:4'''
        g = reference.DirectedGraph(5)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        g.add_edge(0, 3)
        g.add_edge(0, 4)
        g.add_edge(3, 4)
        g.add_edge(4, 3)
        return g
    
    @staticmethod
    def graph_15_2():
        '''This method should construct and return a DirectedGraph encoding example 15.2
        Use the following indexes: A:0, B:1, C:2, A':3, B':4, C':5'''
        g = reference.DirectedGraph(6)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        g.add_edge(3, 4)
        g.add_edge(4, 5)
        g.add_edge(5, 3)
        return g
    
    @staticmethod
    def extra_graph_1():
        '''This method should construct and return a DirectedGraph of your choice with at least 10 nodes'''    
        g = reference.DirectedGraph(15)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 4)
        g.add_edge(4, 5)
        g.add_edge(5, 6)
        g.add_edge(6, 7)
        g.add_edge(7, 8)
        g.add_edge(8, 9)
        g.add_edge(10, 9)
        g.add_edge(11, 9)
        g.add_edge(12, 9)
        g.add_edge(13, 9)
        g.add_edge(14, 9)
        return g
    
    @staticmethod
    def extra_graph_2():
        '''This method should construct and return a DirectedGraph of your choice with at least 10 nodes'''    
        g = reference.DirectedGraph(10)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        g.add_edge(0, 4)
        g.add_edge(0, 5)
        g.add_edge(0, 6)
        g.add_edge(0, 7)
        g.add_edge(0, 8)
        g.add_edge(1, 9)
        return g

    @staticmethod
    def facebook_graph(filename = "datasets/facebook_combined.txt"):
        ''' This method should return a DIRECTED version of the facebook graph as an instance of the DirectedGraph class.
        In particular, if u and v are friends, there should be an edge between u and v and an edge between v and u.'''
        with open(filename, mode="r") as f:
            content = f.readlines()
        content = [x.strip().split(' ') for x in content]

        g = reference.DirectedGraph(4039)
        for edge in content:
            g.add_edge(origin_node=int(edge[0]), destination_node=int(edge[1]))
            g.add_edge(origin_node=int(edge[1]), destination_node=int(edge[0]))

        return g

    @staticmethod
    def question8b():
        # Load Facebook graph
        filepath = "datasets/facebook_combined.txt"
        FB_g = reference.facebook_graph(filename = filepath)

        # Run PR for 20 iterations
        pr = reference.scaled_page_rank(graph = FB_g, num_iter=20, enforce_sink=True)

        return pr

####################################################################################################
########### HW4 SUPPLEMENTARY FUNCTIONS ###########################################################
####################################################################################################
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

####################################################################################################
########### UNIT TESTS #############################################################################
####################################################################################################
def compare_ref_DG_to_DG(ref_G : reference.DirectedGraph, G : DirectedGraph):
    for i in range(G.number_of_nodes()):
        for j in range(G.number_of_nodes()):
            if ref_G.check_edge(i, j) != G.get_edge(i, j):
                return False
    return True

class TestPageRankGraphs(unittest.TestCase):

    def setUp(self):
        self.graph_pairs = [(graph_15_1_left(), reference.graph_15_1_left(), "graph_15_1_left"), (graph_15_1_right(), reference.graph_15_1_right(), "graph_15_1_right"), (graph_15_2(), reference.graph_15_2(), "graph_15_2")]

    def test_graphs(self):
        for G, ref_G, name in self.graph_pairs:
            self.assertTrue(compare_ref_DG_to_DG(ref_G, G), f"[Compare Reference to Our Graphs][Test {name}] Failed")
            print(f"[Compare Reference to Our Graphs][Test {name}] Passed")

def create_random_directed_graph(n : int, threshold : float = 0.5):
    G = DirectedGraph(n)
    for i in range(n):
        for j in range(n):
            if i != j and random.random() > threshold:
                G.add_edge(i, j)
    return G

def reference_DG_to_DG(ref_G : reference.DirectedGraph) -> DirectedGraph:
    G = DirectedGraph(ref_G.number_of_nodes())
    for i in range(ref_G.number_of_nodes()):
        for j in ref_G.edges_from(i):
            G.add_edge(i, j)
    return G

def DG_to_reference_DG(G : DirectedGraph) -> reference.DirectedGraph:
    ref_G = reference.DirectedGraph(G.number_of_nodes())
    for i in range(G.number_of_nodes()):
        for j in G.edges_from(i):
            ref_G.add_edge(i, j)
    return ref_G

def compare_scores(our_scores, ref_scores):
    for i in range(len(our_scores)):
        if abs(our_scores[i] - ref_scores[i]) > 1e-5:
            return False
    return True

class TestPageRank(unittest.TestCase):

    def setUp(self):
        # Our graphs
        self.our_graphs = [graph_15_1_left(), graph_15_1_right(), graph_15_2(), extra_graph_1(), extra_graph_2()]
        self.ref_graphs = [reference.graph_15_1_left(), reference.graph_15_1_right(), reference.graph_15_2(), reference.extra_graph_1(), reference.extra_graph_2()]

        # Random graphs with no sinks
        self.random_graphs_no_sinks = [create_random_directed_graph(20, 0.5) for _ in range(20)] + [create_random_directed_graph(30, 0.8) for _ in range(20)]
        for graph in self.random_graphs_no_sinks:
            for i in range(graph.number_of_nodes()):
                if len(graph.edges_from(i)) == 0:
                    graph.add_edge(i, i)

        # Random graphs that may have sinks
        self.random_graphs = [create_random_directed_graph(20, 0.5) for _ in range(20)] + [create_random_directed_graph(30, 0.8) for _ in range(20)]

        # Random graphs with sinks
        for graph_ind in range(30):
            G = DirectedGraph(20)
            rank_sink = random.randint(0, G.number_of_nodes() - 1)
            for i in range(G.number_of_nodes()):
                if i != rank_sink:
                    for j in range(G.number_of_nodes()):
                        if i != j and random.random() > 0.5:
                            G.add_edge(i, j)
            self.random_graphs.append(G)

        # Configurations
        self.configs = list(product(list(range(10,11)), [0.1, 1 / 7, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))

    def test_our_graphs(self):
        for test_i, G in enumerate(self.our_graphs):
            for config in self.configs:
                i, eps = config
                our_scores = scaled_page_rank(G, i, eps)
                ref_scores = reference.scaled_page_rank(DG_to_reference_DG(G), i, eps, enforce_sink=True)

                # Verify that the scores are the same
                self.assertTrue(compare_scores(our_scores, ref_scores), f"[Our Graphs][Test {test_i}] Failed: {i=} {eps=} {our_scores=} {ref_scores=}")

                # Verify that the ranks are indeed valid ranks
                self.assertAlmostEqual(sum(our_scores.values()), 1.0, places=5, msg=f"[Our Graphs][Test {test_i}] L1 Norm of Scores is not 1")
                self.assertTrue(all(0 <= score <= 1 for score in our_scores.values()))
                
                # Print that the test passed
                print(f"[Our Graphs][Test {test_i}] Passed: {i=} {eps=} {our_scores=} {ref_scores=}")

    def test_ref_graphs(self):
        for test_i, G in enumerate(self.ref_graphs):
            for config in self.configs:
                i, eps = config
                our_scores = scaled_page_rank(reference_DG_to_DG(G), i, eps)
                ref_scores = reference.scaled_page_rank(G, i, eps, enforce_sink=True)

                # Verify that the scores are the same
                self.assertTrue(compare_scores(our_scores, ref_scores), f"[Ref Graphs][Test {test_i}] Failed: {i=} {eps=} {our_scores=} {ref_scores=}")

                # Verify that the ranks are indeed valid ranks
                self.assertAlmostEqual(sum(our_scores.values()), 1.0, places=5, msg=f"[Ref Graphs][Test {test_i}] L1 Norm of Scores is not 1")
                self.assertTrue(all(0 <= score <= 1 for score in our_scores.values()))

                # Print that the test passed
                print(f"[Ref Graphs][Test {test_i}] Passed: {i=} {eps=} {our_scores=} {ref_scores=}")

    def test_random_graphs_no_sinks(self):
        for test_i, G in enumerate(self.random_graphs_no_sinks):
            for config in self.configs:
                i, eps = config
                our_scores = scaled_page_rank(G, i, eps)
                ref_scores = reference.scaled_page_rank(DG_to_reference_DG(G), i, eps)

                # Verify that the scores are the same
                self.assertTrue(compare_scores(our_scores, ref_scores), f"[Random Graphs No Sinks][Test {test_i}] Failed: {i=} {eps=}")

                # Verify that the ranks are indeed valid ranks
                self.assertAlmostEqual(sum(our_scores.values()), 1.0, places=5, msg=f"[Random Graphs No Sinks][Test {test_i}] L1 Norm of Scores is not 1")
                self.assertTrue(all(0 <= score <= 1 for score in our_scores.values()))

                # Print that the test passed
                print(f"[Random Graphs No Sinks][Test {test_i}] Passed: {i=} {eps=}")

    def test_random_graphs_may_have_sinks(self):
        for test_i, G in enumerate(self.random_graphs):
            for config in self.configs:
                i, eps = config
                our_scores = scaled_page_rank(G, i, eps)
                ref_scores = reference.scaled_page_rank(DG_to_reference_DG(G), i, eps, enforce_sink=True)

                # Verify that the scores are the same
                self.assertTrue(compare_scores(our_scores, ref_scores), f"[Random Graphs May Have Sinks][Test {test_i}] Failed: {i=} {eps=}")

                # Verify that the ranks are indeed valid ranks
                self.assertAlmostEqual(sum(our_scores.values()), 1.0, places=5, msg=f"[Random Graphs May Have Sinks][Test {test_i}] L1 Norm of Scores is not 1")
                self.assertTrue(all(0 <= score <= 1 for score in our_scores.values()))

                # Print that the test passed
                print(f"[Random Graphs May Have Sinks][Test {test_i}] Passed: {i=} {eps=}, Sink Present={any(len(G.edges_from(i)) == 0 for i in range(G.number_of_nodes()))}")

class TestProblem8(unittest.TestCase):

    def test_same_graph(self):
        reference_facebook_graph = reference.facebook_graph()
        our_facebook_graph = facebook_graph()
        self.assertTrue(compare_ref_DG_to_DG(reference_facebook_graph, our_facebook_graph), f"[Facebook Graph Comparison to Reference] Failed")
        print(f"[Facebook Graph Comparison to Reference] Passed")

    def test_facebook_graph(self):
        # Load Facebook graph
        our_facebook_graph = facebook_graph()

        # Verify the edges from and edges to are equal (as the graph is undirected)
        for i in range(our_facebook_graph.number_of_nodes()):
            self.assertTrue(set(our_facebook_graph.edges_from(i)) == set(our_facebook_graph.edges_to(i)), f"[Facebook Graph Edges From and To] Failed")

    def test_page_rank(self):
        # Calculate reference ranks
        reference_ranks = reference.question8b()

        # Calculate our ranks
        fb_graph = facebook_graph()
        ranks = scaled_page_rank(fb_graph, 25)

        # Compare the ranks
        self.assertTrue(compare_scores(ranks, reference_ranks), f"[Facebook PageRank Comparison to Reference] Failed")

        # Verify that the ranks are indeed valid ranks
        self.assertAlmostEqual(sum(ranks.values()), 1.0, places=5, msg=f"[Facebook PageRank Comparison to Reference] L1 Norm of Scores is not 1")
        self.assertTrue(all(0 <= score <= 1 for score in ranks.values()))

        # Print that the test passed
        print(f"[Facebook PageRank Comparison to Reference] Passed")

if __name__ == '__main__':
    unittest.main()


#### RECYCLE BIN OF WORK #######
# def plot_facebook_ranks(fb_graph: DirectedGraph, ranks: dict):
#     """This method should plot the facebook graph using the plot_graph method."""
#     from matplotlib import cm
#     import networkx as nx
#     import matplotlib.pyplot as plt
#     import numpy as np

#     # Create a colormap
#     viridis = cm.get_cmap('RdYlBu')

#     # Create a directed NetworkX graph
#     network = nx.DiGraph()
#     network.add_nodes_from(np.arange(fb_graph.number_of_nodes()))

#     for node in range(fb_graph.number_of_nodes()):
#         for to in fb_graph.edges_from(node):
#             network.add_edge(node, to)

#     # Set node colors to red at the highest rank (200), white at the lowest rank (200), and orange in between (rest)
#     colors = np.full((fb_graph.number_of_nodes()), ('#808080'))
#     sorted_rank = np.argsort(list(ranks.values()))
#     for rank in sorted_rank[-200:]:
#         colors[rank] = '#FF0000' # red

#     # Adjust the figure size
#     plt.figure(figsize=(12, 10))  # Increase the figure size

#     # Draw the graph
#     nx.draw(network, node_size=10, width=0.05, with_labels=False,
#             node_color=colors, cmap=viridis)

#     plt.savefig("figures/facebook.png", format="png")

# # 8 (a): extract facebook graph
# fb_graph = facebook_graph()

# # 8 (b): run scaled page rank on facebook graph
# ranks = scaled_page_rank(fb_graph, 20)
# plot_facebook_ranks(fb_graph, ranks)