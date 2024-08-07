import networkx as nx
import random
import unittest

from hw2_p10 import *

def max_matching_special(n, m, C):
    """Given n drivers, m riders, and a set of matching constraints C,
    output a maximum matching. Specifically, C is a n x m array, where
    C[i][j] = 1 if driver i (in 0...n-1) and rider j (in 0...m-1) are compatible.
    If driver i and rider j are incompatible, then C[i][j] = 0.
    Return an n-element array M where M[i] = j if driver i is matched with rider j,
    and M[i] = None if driver i is not matched."""
    match_graph = WeightedDirectedGraph(n + m + 2)
    source = 0
    sink = n + m + 1
    for i in range(n):
        match_graph.set_edge(source, i + 1, 1)
        for j in filter(lambda j: C[i][j], range(m)):
            match_graph.set_edge(i + 1, n + j + 1, 1)
    for j in range(m):
        match_graph.set_edge(n + j + 1, sink, 1)
    flow_val, F = max_flow(match_graph, source, sink)

    # Extract the matching
    matching = []
    for i in range(n):
        matching.append(None)
        for j in range(m):
            if F.get_edge(i + 1, n + j + 1) == 1:
                matching[i] = j
                break

    # Return the matching
    return matching, F

def create_random_directed_graph(node_amount=10, thresh=0.5, capacity_limit=10):

    # create the networkx version of the graph
    G = nx.DiGraph()
    for i in range(node_amount):
        G.add_node(i)
    for i in range(node_amount):
        for j in range(node_amount):
            if random.random() < thresh:
                G.add_edge(i, j, capacity=random.randint(0, capacity_limit))

    # create the WDG version of the graph
    WDG = WeightedDirectedGraph(node_amount)
    for i in range(node_amount):
        for j in range(node_amount):
            if G.has_edge(i, j):
                WDG.set_edge(i, j, G[i][j]['capacity'])

    # return both
    return G, WDG

def create_random_bipartite_graph(left_side_node_amount=10, right_side_node_amount=10, thresh=0.2):
    
    # Create constraints matrix
    C = [[0 if random.random() < thresh else 1 for _ in range(right_side_node_amount)] for _ in range(left_side_node_amount)]

    # create the networkx version of the graph
    G = nx.Graph()
    G.add_nodes_from(list(range(left_side_node_amount)), bipartite=0)
    G.add_nodes_from(list(range(left_side_node_amount, left_side_node_amount + right_side_node_amount)), bipartite=1)

    for i in range(left_side_node_amount):
        for j in range(right_side_node_amount):
            if C[i][j] == 1:
                G.add_edge(i, j + left_side_node_amount)

    # return both
    return G, C

class TestMaxFlowRobust(unittest.TestCase):

    def setUp(self):
        self.test_amount = 200
        self.node_amount = 50
        self.thresh = 0.5
        self.capacity_limit = 40
        self.random_graphs = [create_random_directed_graph(self.node_amount, self.thresh, self.capacity_limit) for _ in range(self.test_amount)]

    def test_max_flow(self):
        
        # calculate max flow value for each graph
        for i, (G, WDG) in enumerate(self.random_graphs):

            # get some random source and sink
            source = random.randint(0, len(G.nodes()) - 1)
            sink = random.choice([i for i in range(len(G.nodes())) if i != source])
                   
            # calculate max flow for G using networkx
            nx_max_flow = nx.maximum_flow_value(G, source, sink)

            # calculate max flow for WDG using our implementation
            WDG_max_flow, F = max_flow(WDG, source, sink)

            # assert that the flow is non-negative and less than the capacity
            self.assertTrue(
                all(
                    0 <= F.get_edge(origin, destination) <= WDG.get_edge(origin, destination)
                    for origin in range(WDG.number_of_nodes())
                    for destination in WDG.edges_from(origin)
                ),
                msg="flow is not non-negative or exceeds capacity"
            )

            # assert conservation of flow
            self.assertTrue(
                all(
                    sum(F.get_edge(origin, destination) for destination in range(WDG.number_of_nodes())) == sum(F.get_edge(destination, origin) for destination in range(WDG.number_of_nodes()))
                    if origin != source and origin != sink
                    else True
                    for origin in range(WDG.number_of_nodes())
                ),
                msg="flow is not conserved"
            )

            # assert flow value is as advertised
            self.assertEqual(
                sum(F.get_edge(source, destination) for destination in WDG.edges_from(source)),
                WDG_max_flow,
                msg="flow value is incorrect"
            )

            # check if they are equal
            self.assertEqual(nx_max_flow, WDG_max_flow)
            print(f"[MaxFlowTest][{i}] Passed: {nx_max_flow} == {WDG_max_flow}")

class TestMaximumMatching(unittest.TestCase):

    def setUp(self):
        self.test_amount = 200
        self.left_side_node_amount = 25
        self.right_side_node_amount = 35
        self.total_node_amount = self.left_side_node_amount + self.right_side_node_amount
        self.thresh = 0.97
        self.random_graphs = [create_random_bipartite_graph(self.left_side_node_amount, self.right_side_node_amount, self.thresh) for _ in range(self.test_amount)]

    def test_maximum_matching(self):
        
        # calculate max flow value for each graph
        for i, (G, C) in enumerate(self.random_graphs):

            # calculate maximum matching for G using networkx
            nx_matching = nx.bipartite.maximum_matching(G, top_nodes=list(range(self.left_side_node_amount)))
            nx_max_matching = len(nx_matching) // 2 # each edge is counted twice

            # calculate maximum matching for WDG using our implementation
            WDG_matching, F = max_matching_special(self.left_side_node_amount, self.right_side_node_amount, C)

            # assert that the flow is non-negative and less than the capacity
            self.assertTrue(
                all(
                    0 <= F.get_edge(origin, destination) <= 1
                    for origin in range(F.number_of_nodes())
                    for destination in range(F.number_of_nodes())
                ),
                msg="flow is not non-negative or exceeds capacity"
            )

            # assert conservation of flow
            self.assertTrue(
                all(
                    sum(F.get_edge(origin, destination) for destination in range(F.number_of_nodes())) == sum(F.get_edge(destination, origin) for destination in range(F.number_of_nodes()))
                    if origin != 0 and origin != F.number_of_nodes() - 1
                    else True
                    for origin in range(F.number_of_nodes())
                ),
                msg=f"flow is not conserved" #, {self.left_side_node_amount=}, {self.right_side_node_amount=}, {C=}"
            )

            # assert flow value is as advertised
            self.assertEqual(
                sum(F.get_edge(0, destination) for destination in range(1, self.left_side_node_amount + 1)),
                len(list(filter(lambda x: x != None, WDG_matching))),
                msg=f"flow value is incorrect, {self.left_side_node_amount=}, {self.right_side_node_amount=}, {C=}, {WDG_matching=}"
            )

            # assert that a vertex has at most one incoming edge
            self.assertTrue(
                all(
                    sum(1 for origin in range(F.number_of_nodes()) if F.get_edge(origin, destination) > 0) <= 1
                    for destination in range(F.number_of_nodes())
                    if destination != 0 and destination != F.number_of_nodes() - 1
                ),
                msg=f"a vertex has more than one incoming edge, {self.left_side_node_amount=}, {self.right_side_node_amount=}, {C=}"
            )

            # assert that a vertex has at most one outgoing edge
            self.assertTrue(
                all(
                    sum(1 for destination in range(F.number_of_nodes()) if F.get_edge(origin, destination) > 0) <= 1
                    for origin in range(F.number_of_nodes())
                    if origin != 0 and origin != F.number_of_nodes() - 1
                ),
                msg="a vertex has more than one outgoing edge"
            )

            # validate matching is indeed contained in the graph
            self.assertTrue(
                all(C[i][WDG_matching[i]] == 1 for i in range(self.left_side_node_amount) if WDG_matching[i] is not None),
                msg="matching is not contained in the graph"
            )

            # Validate matching is indeed a matching
            assert len(set(filter(lambda x: x != None, WDG_matching))) == len(list(filter(lambda x: x != None, WDG_matching))), f"[max_matching]: matching is not a matching as two riders are matched with the same driver, {WDG_matching=}, {n=}, {C=}"

            # check if they are equal
            self.assertEqual(nx_max_matching, len(list(filter(lambda x: x != None, WDG_matching))))
            print(f"[MaxMatchingTest][{i}] Passed: {nx_max_matching} == {len(list(filter(lambda x: x != None, WDG_matching)))}")

class TestWeightedDirectedGraph(unittest.TestCase):

    def setUp(self):
        self.graph = WeightedDirectedGraph(5)

    def test_initialization(self):
        self.assertEqual(self.graph.number_of_nodes(), 5)
        self.assertEqual(self.graph.edges, {n: dict() for n in range(5)})

    def test_set_edge_add(self):
        self.graph.set_edge(0, 1, 3)
        self.assertEqual(self.graph.get_edge(0, 1), 3)
        self.assertIn(1, self.graph.edges_from(0))

    def test_set_edge_modify(self):
        self.graph.set_edge(0, 1, 3)
        self.graph.set_edge(0, 1, 5)
        self.assertEqual(self.graph.get_edge(0, 1), 5)

    def test_set_edge_remove(self):
        self.graph.set_edge(0, 1, 3)
        self.graph.set_edge(0, 1, 0)
        self.assertEqual(self.graph.get_edge(0, 1), 0)
        self.assertNotIn(1, self.graph.edges_from(0))

    def test_edges_from(self):
        self.graph.set_edge(0, 1, 3)
        self.graph.set_edge(0, 2, 4)
        self.assertEqual(self.graph.edges_from(0), [1, 2])
        self.assertEqual(self.graph.edges_from(1), [])
        self.graph.set_edge(0, 1, 0)
        self.assertEqual(self.graph.edges_from(0), [2])

    def test_get_edge(self):
        self.graph.set_edge(0, 1, 3)
        self.assertEqual(self.graph.get_edge(0, 1), 3)
        self.assertEqual(self.graph.get_edge(0, 2), 0)

    def test_number_of_nodes(self):
        self.assertEqual(self.graph.number_of_nodes(), 5)


class TestWeightedDirectedGraph(unittest.TestCase):

    def setUp(self):
        self.graph = WeightedDirectedGraph(5)

    def test_initialization(self):
        self.assertEqual(self.graph.number_of_nodes(), 5)
        self.assertEqual(self.graph.edges, {n: dict() for n in range(5)})

    def test_set_edge_add(self):
        self.graph.set_edge(0, 1, 3)
        self.assertEqual(self.graph.get_edge(0, 1), 3)
        self.assertIn(1, self.graph.edges_from(0))

    def test_set_edge_modify(self):
        self.graph.set_edge(0, 1, 3)
        self.graph.set_edge(0, 1, 5)
        self.assertEqual(self.graph.get_edge(0, 1), 5)

    def test_set_edge_remove(self):
        self.graph.set_edge(0, 1, 3)
        self.graph.set_edge(0, 1, 0)
        self.assertEqual(self.graph.get_edge(0, 1), 0)
        self.assertNotIn(1, self.graph.edges_from(0))

    def test_edges_from(self):
        self.graph.set_edge(0, 1, 3)
        self.graph.set_edge(0, 2, 4)
        self.assertEqual(self.graph.edges_from(0), [1, 2])
        self.assertEqual(self.graph.edges_from(1), [])
        self.graph.set_edge(0, 1, 0)
        self.assertEqual(self.graph.edges_from(0), [2])

    def test_get_edge(self):
        self.graph.set_edge(0, 1, 3)
        self.assertEqual(self.graph.get_edge(0, 1), 3)
        self.assertEqual(self.graph.get_edge(0, 2), 0)

    def test_number_of_nodes(self):
        self.assertEqual(self.graph.number_of_nodes(), 5)


class TestWeightedDirectedGraphBFS(unittest.TestCase):

    def setUp(self):
        self.graph = WeightedDirectedGraph(6)
        # Construct a graph for testing
        self.graph.set_edge(0, 1, 1)
        self.graph.set_edge(0, 2, 1)
        self.graph.set_edge(1, 3, 1)
        self.graph.set_edge(2, 4, 1)
        self.graph.set_edge(3, 5, 1)
        self.graph.set_edge(4, 5, 1)

    def test_bfs_path_exists(self):
        parent = [-1] * self.graph.number_of_nodes()
        self.assertFalse(self.graph.single_source_bfs(0))
        self.assertTrue(self.graph.distances[0][5] >= 0)  # Ensure there is a path from 0 to 5

    def test_bfs_no_path(self):
        self.assertFalse(self.graph.single_source_bfs(1))
        self.assertEqual(self.graph.distances[1][4], -1)  # Ensure the distance is -1

    def test_bfs_single_node(self):
        single_node_graph = WeightedDirectedGraph(1)
        self.assertFalse(single_node_graph.single_source_bfs(0))
        self.assertEqual(single_node_graph.distances[0][0], 0) # Ensure the distance to itself is 0

    def test_bfs_disconnected_graph(self):
        self.graph.set_edge(4, 5, 0)  # Remove edge to disconnect part of the graph
        self.assertFalse(self.graph.single_source_bfs(4))
        self.assertTrue(self.graph.distances[4][5] == -1) # Ensure node 5 is not reachable from node 4

class Robust(unittest.TestCase):
    def test_small_graph(self):
        G = WeightedDirectedGraph(4)
        G.set_edge(0, 1, 1)
        G.set_edge(0, 2, 3)
        G.set_edge(1, 2, 2)
        G.set_edge(1, 3, 1)
        G.set_edge(2, 3, 1)

        s, t = 0, 3
        max_flow_value, flow_graph = max_flow(G, s, t)

        self.assertEqual(max_flow_value, 2)

    def test_medium_graph(self):
        G = WeightedDirectedGraph(6)
        G.set_edge(0, 1, 16)
        G.set_edge(0, 2, 13)
        G.set_edge(1, 2, 10)
        G.set_edge(1, 3, 12)
        G.set_edge(2, 1, 4)
        G.set_edge(2, 4, 14)
        G.set_edge(3, 2, 9)
        G.set_edge(3, 5, 20)
        G.set_edge(4, 3, 7)
        G.set_edge(4, 5, 4)

        s, t = 0, 5
        max_flow_value, flow_graph = max_flow(G, s, t)

        self.assertEqual(max_flow_value, 23)

        expected_flows = {
            (0, 1): 12,
            (1, 0): -12,
            (0, 2): 11,
            (2, 0): -11,
            (1, 2): 0,
            (2, 1): 0,
            (1, 3): 12,
            (3, 1): -12,
            (2, 4): 11,
            (4, 2): -11,
            (3, 2): 1,
            (2, 3): -1,
            (3, 5): 19,
            (5, 3): -19,
            (4, 3): 7,
            (3, 4): -7,
            (4, 5): 4,
            (5, 4): -4,
        }

        # verify flow values
        expected_flow_value = sum([expected_flows.get((0, v), 0) for v in range(6)])
        self.assertEqual(max_flow_value, expected_flow_value)

        # verify returned flow is valid
        self.assertTrue(all([flow_graph.get_edge(u, v) >= 0 for u in range(6) for v in range(6)])) # non-negative
        self.assertTrue(all([flow_graph.get_edge(u, v) <= G.get_edge(u, v) for u in range(6) for v in range(6)])) # capacity constraint
        for v in range(6): # flow conservation (except for source and sink)
            if v != s and v != t:
                self.assertEqual(sum([flow_graph.get_edge(u, v) for u in range(6)]), sum([flow_graph.get_edge(v, u) for u in range(6)]))

        # for u in range(6):
        #     for v in range(6):
        #         print(u, v, flow_graph.get_edge(u, v))
        #         self.assertEqual(flow_graph.get_edge(u, v), expected_flows.get((u, v), 0))

    def test_disconnected_graph(self):
        G = WeightedDirectedGraph(4)
        G.set_edge(0, 1, 3)
        G.set_edge(2, 3, 2)

        s, t = 0, 3
        max_flow_value, flow_graph = max_flow(G, s, t)

        self.assertEqual(max_flow_value, 0)
        self.assertEqual(flow_graph.edges, {n: dict() for n in range(4)})

    def test_no_edges(self):
        G = WeightedDirectedGraph(4)

        s, t = 0, 3
        max_flow_value, flow_graph = max_flow(G, s, t)

        self.assertEqual(max_flow_value, 0)
        self.assertEqual(flow_graph.edges, {n: dict() for n in range(4)})


class TestMaxMatching(unittest.TestCase):

    def test_basic_matching(self):
        # Test a basic case where all drivers and riders are compatible
        n = 3
        m = 2
        C = [
            [1, 1],
            [1, 1],
            [1, 1]
        ]
        expected_matching = [0, 1, None]  # Drivers 0 and 1 matched with riders 0 and 1 respectively

        result = max_matching(n, m, C)
        self.assertEqual(result, expected_matching)

    def test_no_matching(self):
        # Test where no drivers and riders are compatible
        n = 3
        m = 2
        C = [
            [0, 0],
            [0, 0],
            [0, 0]
        ]
        expected_matching = [None, None, None]  # No drivers matched with any riders

        result = max_matching(n, m, C)
        self.assertEqual(result, expected_matching)

    def test_partial_matching(self):
        # Test where only some drivers and riders are compatible
        n = 3
        m = 2
        C = [
            [1, 0],
            [0, 1],
            [1, 0]
        ]
        expected_matching = [0, 1, None]  # Drivers 0 and 1 matched with riders 0 and 1 respectively

        result = max_matching(n, m, C)
        self.assertEqual(result, expected_matching)

    def test_large_matching(self):
        # Test with a larger number of drivers and riders
        n = 5
        m = 4
        C = [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1]
        ]
        expected_matching = [0, 1, 2, None, 3]  # Example expected matching

        result = max_matching(n, m, C)
        self.assertEqual(result, expected_matching)

    def test_empty_input(self):
        # Test with empty input
        n = 0
        m = 0
        C = []
        expected_matching = []  # No drivers and riders

        result = max_matching(n, m, C)
        self.assertEqual(result, expected_matching)

    def test_large_numbers(self):
        # Test with large numbers of drivers and riders
        n = 100
        m = 50
        # Assume all pairs are compatible for simplicity
        C = [[1] * m for _ in range(n)]
        expected_matching = list(range(m)) + [None] * (n - m)  # All drivers matched with all riders

        result = max_matching(n, m, C)
        self.assertEqual(result, expected_matching)


class TestRandomDriverRiderBipartiteGraph(unittest.TestCase):

    def test_graph_generation(self):
        # Test the basic case with n = 3 and p = 0.5
        n = 3
        p = 0.5
        C = random_driver_rider_bipartite_graph(n, p)

        # Check dimensions
        self.assertEqual(len(C), n)
        for row in C:
            self.assertEqual(len(row), n)

        # Check values are either 0 or 1
        for row in C:
            for value in row:
                self.assertIn(value, [0, 1])

    def test_edge_cases(self):
        # Test edge cases: n = 0, p = 0 and p = 1
        n_values = [0, 1, 5, 10, 100]
        p_values = [0, 1]

        for n in n_values:
            for p in p_values:
                C = random_driver_rider_bipartite_graph(n, p)

                # Check dimensions
                self.assertEqual(len(C), n)
                for row in C:
                    self.assertEqual(len(row), n)

                # Check all values are either 0 or 1
                for row in C:
                    for value in row:
                        self.assertEqual(value, p)


if __name__ == '__main__':
    unittest.main()
