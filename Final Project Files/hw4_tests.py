import random
import unittest
from itertools import product
from hw4 import *

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
    def scaled_page_rank(graph, num_iter, eps=1/7.0):
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

####################################################################################################
########### UNIT TESTS #############################################################################
####################################################################################################
def create_random_directed_graph(n : int, threshold : float = 0.5):
    G = DirectedGraph(n)
    for i in range(n):
        for j in range(n):
            if i != j and random.random() > threshold:
                G.add_edge(i, j)
    return G

class TestPageRankGraphs(unittest.TestCase):

    def setUp(self):
        self.graph_pairs = [(graph_15_1_left(), reference.graph_15_1_left()), (graph_15_1_right(), reference.graph_15_1_right()), (graph_15_2(), reference.graph_15_2())]

    def test_graphs(self):
        for G, ref_G in self.graph_pairs:
            for i in range(G.number_of_nodes()):
                for j in range(G.number_of_nodes()):
                    self.assertEqual(G.get_edge(i, j), ref_G.check_edge(i, j))

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
        self.our_graphs = [graph_15_1_left(), graph_15_1_right(), graph_15_2(), extra_graph_1(), extra_graph_2()]
        self.ref_graphs = [reference.graph_15_1_left(), reference.graph_15_1_right(), reference.graph_15_2(), reference.extra_graph_1(), reference.extra_graph_2()]
        self.random_graphs = [create_random_directed_graph(20, 0.5) for _ in range(20)] + [create_random_directed_graph(30, 0.8) for _ in range(20)]
        self.configs = list(product(list(range(10,11)), [0.1, 1 / 7, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))

    def test_our_graphs(self):
        for test_i, G in enumerate(self.our_graphs):
            for config in self.configs:
                i, eps = config
                our_scores = scaled_page_rank(G, i, eps)
                ref_scores = reference.scaled_page_rank(DG_to_reference_DG(G), i, eps)
                self.assertTrue(compare_scores(our_scores, ref_scores))
                print(f"[Our Graphs][Test {test_i}] {i=} {eps=} {our_scores=} {ref_scores=}")

    def test_ref_graphs(self):
        for test_i, G in enumerate(self.ref_graphs):
            for config in self.configs:
                i, eps = config
                our_scores = scaled_page_rank(reference_DG_to_DG(G), i, eps)
                ref_scores = reference.scaled_page_rank(G, i, eps)
                self.assertTrue(compare_scores(our_scores, ref_scores))
                print(f"[Ref Graphs][Test {test_i}] {i=} {eps=} {our_scores=} {ref_scores=}")

    def test_random_graphs(self):
        for test_i, G in enumerate(self.random_graphs):
            for config in self.configs:
                i, eps = config
                our_scores = scaled_page_rank(G, i, eps)
                ref_scores = reference.scaled_page_rank(DG_to_reference_DG(G), i, eps)
                self.assertTrue(compare_scores(our_scores, ref_scores))
                print(f"[Random Graphs][Test {test_i}] {i=} {eps=}")

if __name__ == '__main__':
    unittest.main()