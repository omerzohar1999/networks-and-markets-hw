import networkx as nx
import random
import unittest
from hw3_matchingmarket import *
from hw3_uber import *

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

class TestMaxFlow(unittest.TestCase):

    def setUp(self):
        self.test_amount = 40
        self.node_amount = 50
        self.thresh = 0.5
        self.capacity_limit = 40
        self.random_graphs = [create_random_directed_graph(self.test_amount, self.thresh, self.capacity_limit) for _ in range(self.node_amount)]

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
        self.test_amount = 40
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
            WDG_matching, _, F = max_matching(self.left_side_node_amount, self.right_side_node_amount, C)

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

class TestMarketEquilibrium(unittest.TestCase):

    def setUp(self):
        pass

    def test_1(self):
        n = 4
        m = 3
        V = [[10, 1, 1], [1, 10, 1], [1, 1, 10], [1, 1, 1]]
        P, M = market_eq(n, m, V)
        print("[sanity_checks_q7b][T1]", P, M)
        self.assertTrue(
            M == [0, 1, 2, None] and P == [1, 1, 1],
            msg="market_eq failed sanity check"
        )
        print("[sanity_checks_q7b][T1] Passed")

    def test_2(self):
        n = 3
        m = 4
        V = [[10, 1, 1, 1], [1, 10, 1, 1], [1, 1, 10, 1]]
        P, M = market_eq(n, m, V)
        print("[sanity_checks_q7b][T2]", P, M)
        self.assertTrue(
            M == [0, 1, 2] and P == [0, 0, 0, 0],
        )
        print("[sanity_checks_q7b][T2] Passed")

    def test_random(self):

        ### Brute-Force Test
        def scipy_max_social_value(n, m, V):
            # converting V to numpy
            V = np.array(V)

            # Use the linear_sum_assignment function to get the indices of the maximum value
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(V, maximize=True)

            # Return the maximum value
            return V[row_ind, col_ind].sum()
        
        # 50 random tests
        for test_i in range(50):

            # Random n and m
            n = random.randint(1, 20)
            m = random.randint(1, 20)

            # Random valuations
            V = np.random.randint(0, 100, size=(n, m))

            # Calculate the market equilibrium
            P, M = market_eq(n, m, V)

            # Validate the matching is indeed a matching
            for i_1 in range(n):
                for i_2 in range(i_1 + 1, n):
                    if M[i_1] == M[i_2] and M[i_1] is not None:
                        self.assertTrue(False, msg=f"market_eq didn't output a matching, {n=}, {m=}, {V=}, {P=}, {M=}")

            # Validate the matching is a perfect matching
            self.assertTrue(
                sum(1 for i in M if i is not None) == min(n, m),
                msg=f"market_eq didn't output a perfect matching, {n=}, {m=}, {V=}, {P=}, {M=}"
            )

            # Validate non-negative prices
            self.assertTrue(
                all(p >= 0 for p in P),
                msg=f"market_eq didn't output non-negative prices, {n=}, {m=}, {V=}, {P=}, {M=}"
            )

            # Calculate the maximum social value
            scipy_max_sv = scipy_max_social_value(n, m, V)

            # Calculate the social value of the market equilibrium
            market_eq_val = social_value(n, m, V, M)

            # Check if the social value of the market equilibrium is the maximum social value
            self.assertTrue(
                market_eq_val == scipy_max_sv,
                msg=f"market_eq didn't output the maximum social value, {n=}, {m=}, {V=}, {P=}, {M=}, {market_eq_val=}, {scipy_max_sv=}"
            )

            # Streamlined test
            self.assertTrue(
                calc_max_social_value(n, m, V) == scipy_max_sv,
                msg=f"calc_max_social_value didn't output the maximum social value, {n=}, {m=}, {V=}, {P=}, {M=}, {market_eq_val=}, {scipy_max_sv=}"
            )

            # Success
            print(f"[sanity_checks_q7b][rand][T{test_i}]: Test passed {market_eq_val} == {scipy_max_sv}")

class TestVCG(unittest.TestCase):

    def setUp(self):
        pass

    def test_1(self):
        n = 4
        m = 3
        V = [[10, 1, 1], [1, 10, 1], [1, 1, 10], [1, 1, 1]]
        P, M = vcg(n, m, V)
        print("[sanity_checks_q7c][T1]", P, M)
        self.assertTrue(
            M == [0, 1, 2, None] and P == [1, 1, 1],
            msg="vcg failed sanity check"
        )
        print("[sanity_checks_q7c][T1] Passed")

    def test_2(self):
        n = 3
        m = 4
        V = [[10, 1, 1, 1], [1, 10, 1, 1], [1, 1, 10, 1]]
        P, M = vcg(n, m, V)
        print("[sanity_checks_q7c][T2]", P, M)
        self.assertTrue(
            M == [0, 1, 2] and P == [0, 0, 0, 0],
        )
        print("[sanity_checks_q7c][T2] Passed")

    def test_random(self):

        ### Brute-Force Test
        def scipy_max_social_value(n, m, V):
            # converting V to numpy
            V = np.array(V)

            # Use the linear_sum_assignment function to get the indices of the maximum value
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(V, maximize=True)

            # Return the maximum value
            return V[row_ind, col_ind].sum()
        
        # 50 random tests
        for test_i in range(50):

            # Random n and m
            n = random.randint(1, 20)
            m = random.randint(1, 20)

            # Random valuations
            V = np.random.randint(0, 100, size=(n, m))

            # Calculate the vcg + clarke pivot rule prices and matching
            P, M = vcg(n, m, V)

            # Validate the matching is indeed a matching
            for i_1 in range(n):
                for i_2 in range(i_1 + 1, n):
                    if M[i_1] == M[i_2] and M[i_1] is not None:
                        self.assertTrue(False, msg=f"vcg didn't output a matching, {n=}, {m=}, {V=}, {P=}, {M=}")

            # Validate the matching is a perfect matching
            self.assertTrue(
                sum(1 for i in M if i is not None) == min(n, m),
                msg=f"vcg didn't output a perfect matching, {n=}, {m=}, {V=}, {P=}, {M=}"
            )

            # Validate non-negative prices
            self.assertTrue(
                all(p >= 0 for p in P),
                msg=f"vcg didn't output non-negative prices, {n=}, {m=}, {V=}, {P=}, {M=}"
            )

            # Calculate the social value of the market equilibrium
            market_eq_val = social_value(n, m, V, M)

            # Calculate the maximum social value
            scipy_max_sv = scipy_max_social_value(n, m, V)

            # Check if the social value of the market equilibrium is the maximum social value
            self.assertTrue(
                market_eq_val == scipy_max_sv,
                msg=f"vcg didn't output the maximum social value, {n=}, {m=}, {V=}, {P=}, {M=}, {market_eq_val=}, {scipy_max_sv=}"
            )

            # Calculate the externality values
            externality_values = np.zeros(n)
            for i in range(n):
                row_i = copy(V[i])
                V[i] = np.zeros(m)
                externality_values[i] = scipy_max_social_value(n, m, V)
                V[i] = row_i

            # Calculate externality prices
            externality_prices = np.zeros(m)
            for i in range(n):
                if M[i] is not None:
                    externality_prices[M[i]] += externality_values[i] - (scipy_max_sv - V[i][M[i]])

            # Check if the prices are indeed externality prices
            self.assertTrue(
                all(P[i] == externality_prices[i] for i in range(m)),
                msg=f"vcg didn't output the externality prices, {n=}, {m=}, {V=}, {P=}, {M=}, {externality_prices=}"
            )

            # print success
            print(f"[sanity_checks_q7c][rand][T{test_i}]: Test passed {market_eq_val} == {scipy_max_sv}")

if __name__ == '__main__':
    unittest.main()