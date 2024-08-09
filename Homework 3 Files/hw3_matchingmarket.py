# Skeleton file for HW3 questions 7 and 8
# =====================================
# IMPORTANT: You are NOT allowed to modify the method signatures
# (i.e. the arguments and return types each function takes).
# We will pass your grade through an autograder which expects a specific format.
# =====================================
from copy import copy

# Do not include any other files or an external package, unless it is one of
# [numpy, pandas, scipy, matplotlib, random]
# please contact us before sumission if you want another package approved.
import random
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
import scipy  # TODO: is this ok?


# find maximum matching, from hw2
class WeightedDirectedGraph:
    def __init__(self, number_of_nodes):
        """Assume that nodes are represented by indices/integers between 0 and number_of_nodes - 1."""
        self.n = number_of_nodes
        self.edges = {node: dict() for node in range(number_of_nodes)}
        self.distances = {}

    def set_edge(self, origin_node, destination_node, weight=1):
        """Modifies the weight for the specified directed edge, from origin to destination node,
        with specified weight (an integer >= 0). If weight = 0, effectively removes the edge from
        the graph. If edge previously wasn't in the graph, adds a new edge with specified weight.
        """
        if weight > 0:
            self.edges[origin_node][destination_node] = weight
        elif destination_node in self.edges[origin_node]:
            self.edges[origin_node].pop(destination_node)
        self.distances.clear()

    def edges_from(self, origin_node):
        """This method shold return a list of all the nodes destination_node such that there is
        a directed edge (origin_node, destination_node) in the graph (i.e. with weight > 0).
        """
        return list(self.edges[origin_node].keys()) if origin_node in self.edges else []
        # if origin_node not in self.edges:
        #     return []
        # return list(self.edges[origin_node].keys())

    def get_edge(self, origin_node, destination_node):
        """This method should return the weight (an integer > 0)
        if there is an edge between origin_node and
        destination_node, and 0 otherwise."""
        return (
            self.edges[origin_node].get(destination_node, 0)
            if origin_node in self.edges
            else 0
        )
        # if origin_node not in self.edges:
        #     return 0
        # return self.edges[origin_node].get(destination_node, 0)

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
            if popped not in self.edges:
                continue
            for neighbor in self.edges[popped]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[popped] + 1
                    parents[neighbor] = popped
                    nodes_to_visit.put(neighbor)

        self.distances[sNode] = distances

    def get_path(self, s, t):
        visited = set()
        distances = [-1] * self.n
        parents = distances.copy()

        visited.add(s)
        distances[s] = 0

        nodes_to_visit = Queue()

        nodes_to_visit.put(s)

        while nodes_to_visit.qsize():
            popped = nodes_to_visit.get()
            if popped == t:
                break
            if popped not in self.edges:
                continue
            for neighbor in self.edges[popped]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[popped] + 1
                    parents[neighbor] = popped
                    nodes_to_visit.put(neighbor)
        if distances[t] == -1:
            return None
        path = [t]
        while path[-1] != s:
            path.append(parents[path[-1]])
        return path[::-1]


def residual_graph(G, F):
    R = WeightedDirectedGraph(G.number_of_nodes())

    # initialize R to be the same as G (capacity graph)
    for origin in range(G.number_of_nodes()):
        for destination in G.edges_from(origin):
            R.set_edge(origin, destination, G.get_edge(origin, destination))

    # update R with the flow graph F
    for origin in range(G.number_of_nodes()):
        for destination in G.edges_from(origin):
            flow = F.get_edge(origin, destination)
            R.set_edge(origin, destination, R.get_edge(origin, destination) - flow)
            R.set_edge(destination, origin, R.get_edge(destination, origin) + flow)

    # for origin in range(G.number_of_nodes()):
    #     for destination in G.edges_from(origin):
    #         capacity = G.get_edge(origin, destination)
    #         flow = F.get_edge(origin, destination)
    #         if flow < capacity:
    #             R.set_edge(origin, destination, capacity - flow)
    #         if flow > 0:
    #             R.set_edge(destination, origin, flow)
    return R


def max_flow(G, s, t):
    """Given a WeightedDirectedGraph G, a source node s, a destination node t,
    compute the (integer) maximum flow from s to t, treating the weights of G as capacities.
    Return a tuple (v, F) where v is the integer value of the flow, and F is a maximum flow
    for G, represented by another WeightedDirectedGraph where edge weights represent
    the final allocated flow along that edge."""

    def path_flow(G, path):
        return min(G.get_edge(path[i], path[i + 1]) for i in range(len(path) - 1))
        # flow = float("inf")
        # for i in range(len(path) - 1):
        #     flow = min(flow, G.get_edge(path[i], path[i + 1]))
        # return flow

    # init F
    F = WeightedDirectedGraph(G.number_of_nodes())
    for origin in range(G.number_of_nodes()):
        for destination in G.edges_from(origin):
            F.set_edge(origin, destination, 0)

    # find augmenting paths, update F
    residual = residual_graph(G, F)
    path = residual.get_path(s, t)
    while path is not None:
        flow = path_flow(residual, path)
        for i in range(len(path) - 1):
            origin = path[i]
            destination = path[i + 1]

            # calc the forward and backward flow additions (NOTE: I think this is true, but I'm not sure)
            forward_flow = min(flow, G.get_edge(origin, destination) - F.get_edge(origin, destination))
            residual_flow = flow - forward_flow
            
            # update the flow graph
            F.set_edge(origin, destination, F.get_edge(origin, destination) + forward_flow)
            F.set_edge(destination, origin, F.get_edge(destination, origin) - residual_flow)

        residual = residual_graph(G, F)
        path = residual.get_path(s, t)

    # calculate v
    v = 0
    for neighbor in F.edges_from(s):
        v += F.get_edge(s, neighbor)
    for origin in range(G.number_of_nodes()):
        if F.get_edge(origin, s) > 0:
            v -= F.get_edge(origin, s)

    return v, F

def max_matching(n, m, C):
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

    # If n == m and the flow value is less than n, there is not perfect matching and there's a constricted set
    constricted_set = None
    if flow_val < n and n == m:
        residual = residual_graph(match_graph, F)
        residual.single_source_bfs(source)
        constricted_set = [i for i in range(n) if residual.distances[source][i + 1] != -1]

    # Return the matching
    return matching, constricted_set, F

def max_weight_matching(C: 'n x m array') -> 'players assignment':
    """return the maximum weight of any matching, where C[i, j] is the weight of the edge"""
    res = scipy.optimize.linear_sum_assignment(-C)
    return res[1]

# === Problem 7(a) ===
def matching_or_cset(n, C):
    """Given a bipartite graph, with 2n vertices, with
    n vertices in the left part, and n vertices in the right part,
    and edge constraints C, output a tuple (b, M) where b is True iff
    there is a matching, and M is either a matching or a constricted set.
    -   Specifically, C is a n x n array, where
        C[i][j] = 1 if left vertex i (in 0...n-1) and right vertex j (in 0...n-1)
        are connected by an edge. If there is no edge between vertices
        (i,j), then C[i][j] = 0.
    -   If there is a perfect matching, return an n-element list M where M[i] = j
        if driver i is matched with rider j.
    -   If there is no perfect matching, return a list M of left vertices
        that comprise a constricted set.
    """
    # match_graph = WeightedDirectedGraph(2 * (n + 1))
    # source = 0
    # sink = 2 * n + 1
    # for i in range(n):
    #     match_graph.set_edge(source, i + 1, 1)
    #     for j in filter(lambda j: C[i][j], range(n)):
    #         match_graph.set_edge(i + 1, n + j + 1, 1)
    # for j in range(n):
    #     match_graph.set_edge(n + j + 1, sink, 1)
    # flow_val, F = max_flow(match_graph, source, sink)
    # if flow_val < n:
    #     residual = residual_graph(match_graph, F)
    #     residual.single_source_bfs(source)
    #     return False, [i for i in range(n) if residual.distances[source][i + 1] != -1]

    matching, constricted_set, F = max_matching(n, n, C)

    if constricted_set is not None:
        return False, constricted_set

    # Extract the matching
    matching = []
    for i in range(n):
        for j in range(n):
            if F.get_edge(i + 1, n + j + 1) == 1:
                matching.append(j)
                break

    # Validate matching is indeed a perfect matching (TODO: remove this)
    assert len(matching) == n, f"[matching_or_cset]: matching is not a perfect matching as it doesn't assign all riders, {matching=}, {n=}, {C=}"

    return True, matching


# === Problem 7(b) ===
def create_constraints(P, V):
    """Given a list of m prices P, and a nxm list of valuations V,
    output a nxm list C of edge constraints.
    -   Specifically, P[j] is the price of item j.
        V[i][j] is the value of item j for buyer i.
    -   C[i][j] = 1 if V[i][j]-P[j] is maximal for buyer i, and
        C[i][j] = 0 otherwise.
    """
    m, n = len(P), len(V)
    C = [[0] * m for _ in range(n)]
    for i in range(n):
        maximal = max(V[i][k] - P[k] for k in range(m))
        # NOTE: if there are no non-negative edges, they are all not acceptable and by definition also not preferred. Claim 9.3 in the notes is incorrect, so we need to beware if this check is desired.
        if maximal < 0: 
            continue
        for j in range(m):
            C[i][j] = 1 if V[i][j] - P[j] == maximal else 0
    return C


def make_square_values(V):
    """Given a list of valuations V of size n x m,
    output an equivalent list of valuations of size max(n,m) x max(n,m).
    -   Specifically, V[i][j] is the value of item j for buyer i.
    -   Return a list V' where V'[i][j] = V[i][j] if i < n and j < m,
        and V'[i][j] = 0 otherwise.
    """
    n = len(V)
    m = len(V[0])
    return [
        [V[i][j] if i < n and j < m else 0 for j in range(max(n, m))]
        for i in range(max(n, m))
    ]


def market_eq(n, m, V):
    """Given a matching market with n buyers and m items, and
    valuations V, output a market equilibrium tuple (P, M)
    of prices P and a matching M.
    -   Specifically, V is an n x m list, where
        V[i][j] is a number representing buyer i's value for item j.
    -   Return a tuple (P, M), where P is an m-element list, where
        P[j] equals the price of item j.
        M is an n-element list, where M[i] = j if buyer i is
        matched with item j, and M[i] = None if there is no matching.
    In sum, buyer i receives item M[i] and pays P[M[i]]."""
    P = [0] * max(n, m)
    M = [None] * max(n, m)
    # Gil: consider using rectangular_valuations func instead?
    V_square = make_square_values(V)
    while True:
        C = create_constraints(P, V_square)
        isPerfect, M = matching_or_cset(max(n, m), C)
        if isPerfect:
            break
        # M is a constrained set
        # increase prices of all of their neighbors by 1
        neighbors = set()
        for i in M:
            for j in range(max(n, m)):
                if C[i][j] == 1:
                    neighbors.add(j)
        for j in neighbors:
            P[j] += 1

        # if all prices are positive, decrease all prices by 1
        if all(P[j] > 0 for j in range(max(n, m))):
            P = [P[j] - 1 for j in range(max(n, m))]

    # Filter out the fictive buyers
    M = M[:n]

    # Filter out the fictive items by 'None' matching
    M = [j if j < m else None for j in M]

    # Filter out the fictive prices and return it with the matching
    return P[:m], M


def rectangular_valuations(V: "n * m") -> "max(n, m) * max(n, m)":
    """Given a list of valuations V of size n x m,
    output an equivalent list of valuations of size max(n,m) x max(n,m).
    -   Specifically, V[i][j] is the value of item j for buyer i.
    -   Return a list V' where V'[i][j] = V[i][j] if i < n and j < m,
        and V'[i][j] = 0 otherwise.
    """
    """TODO: I assume non-negative valuations!!"""
    n, m = len(V), len(V[0])
    # min_value = min(V[i][j] for i in range(n) for j in range(m)) - 1
    rec_V = np.zeros((max(n, m), max(n, m)))
    rec_V[:n, :m] = V
    return rec_V

def lec5_page7_example_q7b():
    n = 3
    m = 3
    V = [[4, 12, 5], [7, 10, 9], [7, 7, 10]]
    P, M = market_eq(n, m, V)
    print("[lec5_page7_example_q7b]", P, M)
    assert M == [1, 0, 2], "[lec5_page7_example_q7b]: M is not as expected"
    assert P == [0, 3, 2], "[lec5_page7_example_q7b]: P is not as expected"

# === Problem 8(a) ===
def social_value(n, m, V, M):
    """Given a matching market with n buyers and m items, and
    valuations V, and a matching M, output the social value of the matching.
    -   Specifically, V is an n x m list, where
        V[i][j] is a number representing buyer i's value for item j.
    -   M is an n-element list, where M[i] = j if buyer i is
        matched with item j, and M[i] = None if there is no matching.
    -   Return a number representing the social value of the matching.
    """
    return sum(V[i][M[i]] if M[i] is not None else 0 for i in range(n))

def calc_max_social_value(n, m, V):
    """Given a matching market with n buyers and m items, and
    valuations V, output the maximum social value of the matching.
    -   Specifically, V is an n x m list, where
        V[i][j] is a number representing buyer i's value for item j.
    -   Return a number representing the maximum social value
        that can be achieved in this matching market.
    This is done by computing the market equilibrium and then computing the social value. We have seen in class that the social value of the market equilibrium is the maximum social value.
    """
    P, M = market_eq(n, m, V)
    return social_value(n, m, V, M)

def vcg(n: 'players', m: 'items', V: 'valuations'):
    """Given a matching market with n buyers, and m items, and
    valuations V as defined in market_eq, output a tuple (P,M)
    of prices P and a matching M as computed using the VCG mechanism
    with Clarke pivot rule.
    V,P,M are defined equivalently as for market_eq. Note that
    P[j] should be positive for every j in 0...m-1. Note that P is
    still indexed by item, not by player!!
    """
    """
    X_opt = "Outcome . max |x| sum v_i (x)"
    pi = "sum (j!=i) v_j (x)"
    hi = "-max |x| (j!=i) v_j (x)"
    pi_ = pi + hi
    """
    """TODO: I assume non-negative valuations!! (NOTE: As in the definitions in the notes)"""
    """TODO: needs work on when n!=m (NOTE: I think opting to use the calc_max_social_value which utilizes the market_eq function for the social value maximization algorithm kinda fixes this issue? Validate what I'm saying is true)"""
    rect_V = rectangular_valuations(V)

    # calc market equilibrium as social value maximizing state (i.e., allocation)
    P, M = market_eq(n, m, V)

    # payments
    maximum_social_value = calc_max_social_value(n, m, rect_V)
    P = np.array([maximum_social_value - V[i][M[i]] if M[i] is not None else 0 for i in range(n)])

    # pivot (maximum social value at externality)
    H = np.zeros((n,))
    for i in range(n):
        row_i = copy(rect_V[i])
        rect_V[i] = [0] * len(rect_V[i])
        H[i] = calc_max_social_value(n, m, rect_V) # the ousted maximum social value
        rect_V[i] = row_i

    # externality copmutations: P - H is non-negative by definition
    C = np.zeros((m,))
    for i in range(n):
        j = M[i] # the item that player i is matched with
        if j is not None:
            C[j] = H[i] - P[i] # externality of player i is its price for its item j = M[i]

    return list(C), list(M)

    # rect_V = rectangular_valuations(V)
    # # matching
    # M = max_weight_matching(rect_V)[:n]
    # # payments
    # sum_value = sum(V[i][M[i]] for i in range(n))
    # P = np.array([sum_value - V[i][M[i]] for i in range(n)])
    # # pivot
    # H = np.zeros((n,))
    # for i in range(n):
    #     row_i = copy(rect_V[i])
    #     rect_V[i] = [0] * m
    #     i_M = max_weight_matching(rect_V)
    #     H[i] = sum(rect_V[i][i_M[i]] for i in range(n))
    #     rect_V[i] = row_i

    # # item (-cost) i.e. positive
    # C = np.zeros((m,))
    # for i in range(n):
    #     j = M[i]
    #     C[i] += H[j] - P[j]

    # return list(C), list(M)

def lec5_page7_example_q8a():
    n = 3
    m = 3
    V = [[4, 12, 5], [7, 10, 9], [7, 7, 10]]
    P, M = vcg(n, m, V)
    print("[lec5_page7_example_q8a]", P, M)
    assert M == [1, 0, 2], "[lec5_page7_example_q8a]: M is not as expected"
    assert P == [0, 3, 2], "[lec5_page7_example_q8a]: P is not as expected"

def q8b_analysis():

    # test array
    tests = [
        [[ 3,  0, 15],
       [11, 12,  8],
       [14, 13, 18],
       [ 3, 18,  5],
       [15,  9,  1],
       [14, 17,  9],
       [14, 17, 13]],
       [[12, 14, 16,  8,  6, 17],
       [11,  7,  9, 19,  1, 11],
       [18, 13, 17, 17,  2, 16],
       [15,  0,  4,  1, 15, 15],
       [ 7,  8,  5, 12, 18, 13],
       [ 7, 19,  8, 12,  4,  1]],
       [[ 8, 11,  0,  3,  6,  7],
       [19, 14, 15, 14, 14, 16],
       [17, 19, 19, 13,  8, 17],
       [ 2, 15,  1, 18, 11, 10],
       [ 8,  9,  7, 15,  6, 10],
       [12, 15, 15,  8,  2,  1]],
       [[ 5,  3,  0,  7, 10,  5, 17,  6, 18,  8],
       [ 5,  4,  6,  9, 15,  9, 17,  2, 10, 14],
       [10, 11, 10,  6,  4, 10, 16, 11, 10,  6],
       [ 2, 19,  4, 12,  5,  8, 12,  0, 11, 11],
       [18,  7, 15, 11,  7,  4,  2,  9,  9,  8],
       [ 5,  2,  2,  5,  1, 12, 13, 18,  8,  1]],
       [[15,  3,  1,  5],
       [11, 11, 16,  5],
       [ 9, 15, 13, 17],
       [15, 11, 10, 16],
       [19,  0, 12,  7],
       [17, 16, 13,  9]]
    ]

    # 5 random tests
    for test_i, test_V in enumerate(tests):
        
        # print test number
        print(f"[q8b_analysis] test {test_i}:")

        # cast to numpy
        test_V = np.array(test_V)

        # get n and m
        n = len(test_V)
        m = len(test_V[0])

        # run market equilibrium
        P_eq, M_eq = market_eq(n, m, test_V)

        # run VCG
        P_vcg, M_vcg = vcg(n, m, test_V)
        
        # print results
        print(f"[q8b_analysis][graph]: {n=} {m=} {test_V=}")
        print(f"[q8b_analysis][market_eq]: {P_eq=}, {M_eq=}")
        print(f"[q8b_analysis][vcg]: {P_vcg=}, {M_vcg=}")
        print(f"[q8b_analysis][different?]: {P_eq != P_vcg}")

# === Bonus Question 2(a) (Optional) ===
def random_bundles_valuations(n, m):
    """Given n buyers, m bundles, generate a matching market context
    (n, m, V) where V[i][j] is buyer i's valuation for bundle j.
    Each bundle j (in 0...m-1) is comprised of j copies of an identical good.
    Each player i has its own value for an individual good; this value is sampled
    uniformly at random from [1, 50] inclusive, for each player"""
    individual_rand_values = np.random.randint(1, 51, size=n)
    V = np.tile(np.arange(1, m + 1), (n, 1)) # bundle i is comprised of (i + 1) copies of an identical good
    return (V.T * individual_rand_values).T

def b2a_analysis():
    # Evaluate on n = m = 20 contexts
    for iter_i in range(4):

        # print test number
        print(f"[b2a_analysis] test {iter_i}:")

        # generate random context
        n = 20
        m = 20
        V = random_bundles_valuations(n, m)

        # run VCG
        P_vcg, M_vcg = vcg(n, m, V)

        # print results
        print(f"[b2a_analysis][graph]: {n=} {m=} {V=}")
        print(f"[b2a_analysis][vcg]: {P_vcg=}, {M_vcg=}")

        # get individual valuations
        V_individual = [l[0] for l in V]

        # get indices of sorted valuations
        V_individual_sorted_indices = list(
            sorted(
                list(range(n)),
                key=lambda i: V_individual[i] + (M_vcg[i] / 1000)
            )
        )

        # get the bundles matched to each buyer sorted by individual valuation of the buyer
        M_vcg_sorted = [M_vcg[i] for i in V_individual_sorted_indices]
        print(f"[b2a_analysis][M_vcg_sorted]: {M_vcg_sorted}")
        print(f"[b2a_analysis][M_vcg_sorted increasing?]: {list(sorted(M_vcg_sorted)) == M_vcg_sorted}")

        # get the vcg prices sorted by individual valuation
        # NOTE: this is the externality for each buyer, sorted by individual valuation
        P_vcg_sorted = [P_vcg[i] for i in M_vcg_sorted]

        # print results
        print(f"[b2a_analysis][V_individual]: {V_individual=}")
        print(f"[b2a_analysis][V_individual_sorted]: {np.sort(V_individual)}")
        print(f"[b2a_analysis][V_individual_sorted_indices]: {np.argsort(V_individual)}")
        print(f"[b2a_analysis][vcg_sorted]: {P_vcg_sorted=}")

        # plot the sorted vcg prices
        plt.figure()
        plt.plot(np.sort(V_individual), P_vcg_sorted, 'o-')
        plt.title(f"VCG prices sorted by individual valuation")
        plt.xlabel("Buyer Valuation")
        plt.ylabel("VCG-Clarke-Pivot Price")
        plt.savefig(f"b2a_analysis_{iter_i}.png", format="png")
        plt.savefig(f"b2a_analysis_{iter_i}.pgf", format="pgf")
        plt.show()

# === Bonus Question 2(b) (optional) ===

def gsp_efficient(n, m, V) -> '(P, M)':
    """here V is the valuation of a single item"""
    sorted_bids = list(sorted(enumerate(V), key=lambda x: x[1], reverse=True)) # (index, bid) pairs sorted by descending single-item valuation (bid)
    M = [None] * n # matching from buyer to bundle
    P = np.zeros((m,), dtype=int) # prices

    for rank, (bidder_ind, _) in enumerate(sorted_bids):
        # if we have more buyers than bundles, we can't sell all bundles
        if rank >= m:
            break

        # assign the bundle to the buyer
        M[bidder_ind] = rank

        # set the price of the bundle to the bid of the next highest bidder
        base_cost = sorted_bids[rank + 1][1] if rank + 1 < n else sorted_bids[rank][1] # if this isn't the lowest bidder, the price is the bid of the next highest bidder, else it's the bid of the lowest bidder
        P[rank] = base_cost * (rank + 1) # (rank + 1) items in bundle (rank)

    # return the prices and the matching
    return list(P), M

def calc_utilities_efficient(V, P, M):
    return [((j + 1) * v - P[j])
            if j is not None else 0
            for j, v in zip(M, V)]

# P, M = gsp_efficient(3, 3, [3, 41, 40])
# print(P, M)
# print(calc_utilities_efficient([3, 40, 40], P, M))

def gsp(n, m, V) -> '(P, M)':
    """Given a matching market for bundles with n buyers, and m bundles, and
    valuations V (for bundles), output a tuple (P, M) of prices P and a
    matching M as computed using GSP."""
    V = [V[i][0] for i in range(n)] # valuation of a single item.
    return gsp_efficient(n, m, V)

def b2b_analysis():
    # Evaluate on n = m = 20 contexts
    for iter_i in range(4):

        # print test number
        print(f"[b2b_analysis] test {iter_i}:")

        # generate random context
        n = 20
        m = 20
        V = random_bundles_valuations(n, m)

        # run VCG
        P_vcg, M_vcg = vcg(n, m, V)

        # print results
        print(f"[b2b_analysis][graph]: {n=} {m=} {V=}")
        print(f"[b2b_analysis][vcg]: {P_vcg=}, {M_vcg=}")

        # get individual valuations
        V_individual = [l[0] for l in V]

        # get indices of sorted valuations
        V_individual_sorted_indices = list(
            sorted(
                list(range(n)),
                key=lambda i: V_individual[i] + (M_vcg[i] / 1000)
            )
        )

        # get the bundles matched to each buyer sorted by individual valuation of the buyer
        M_vcg_sorted = [M_vcg[i] for i in V_individual_sorted_indices]
        print(f"[b2b_analysis][M_vcg_sorted]: {M_vcg_sorted}")
        print(f"[b2b_analysis][M_vcg_sorted increasing?]: {list(sorted(M_vcg_sorted)) == M_vcg_sorted}")

        # get the vcg prices sorted by individual valuation
        # NOTE: this is the externality for each buyer, sorted by individual valuation
        P_vcg_sorted = [P_vcg[i] for i in M_vcg_sorted]

        # print results
        print(f"[b2b_analysis][V_individual]: {V_individual=}")
        print(f"[b2b_analysis][V_individual_sorted]: {np.sort(V_individual)}")
        print(f"[b2b_analysis][V_individual_sorted_indices]: {np.argsort(V_individual)}")
        print(f"[b2b_analysis][vcg_sorted]: {P_vcg_sorted=}")

        # run GSP
        P_gsp, M_gsp = gsp(n, m, V)

        # print results
        print(f"[b2b_analysis][gsp]: {P_gsp=}, {M_gsp=}")

        # get the bundles matched to each buyer sorted by individual valuation of the buyer
        M_gsp_sorted = [M_gsp[i] for i in V_individual_sorted_indices]
        print(f"[b2b_analysis][M_gsp_sorted]: {M_gsp_sorted}")
        print(f"[b2b_analysis][M_gsp_sorted increasing?]: {list(sorted(M_gsp_sorted)) == M_gsp_sorted}")

        # get the gsp prices sorted by individual valuation
        P_gsp_sorted = [P_gsp[i] for i in M_gsp_sorted]

        # print results
        print(f"[b2b_analysis][gsp_sorted]: {P_gsp_sorted=}")
        print(f"[b2b_analysis][different?]: {P_vcg_sorted != P_gsp_sorted}")

        # plot the sorted vcg prices and gsp prices
        plt.figure()
        plt.plot(np.sort(V_individual), P_vcg_sorted, 'o-', label="VCG")
        plt.plot(np.sort(V_individual), P_gsp_sorted, 'o-', label="GSP")
        plt.title(f"VCG and GSP prices sorted by individual valuation")
        plt.xlabel("Buyer Valuation")
        plt.ylabel("Price")
        plt.legend()
        # plt.savefig(f"b2b_analysis_{iter_i}.png", format="png")
        # plt.savefig(f"b2b_analysis_{iter_i}.pgf", format="pgf")
        plt.show()

def brd_on_gsp(n, m, V) -> 'V_':
    V = np.array([V[i][1] for i in range(n)])
    real_utilities = lambda V_lie: \
        calc_utilities_efficient(V, *gsp_efficient(n, m, V_lie))

    agents = np.arange(n)
    V_ = np.random.randint(0, V + 1)
    while True:
        np.random.shuffle(agents)
        curr_utilities = real_utilities(V_)
        for i in agents:
            original_val = V_[i]

            best_utility = curr_utilities[i]
            best_val = original_val
            for v in range(V[i] + 1):
                V_[i] = v
                my_utility = real_utilities(V_)[i]
                if my_utility > best_utility:
                    best_utility, best_val = my_utility, v

            if original_val != best_val:
                V_[i] = best_val
                break

            V_[i] = original_val
            continue
        else:   # did not break -> no change
            break
    return V_

# V = [[0, 10], [0, 20], [0, 30], [0, 31]]
# V_real = [l[1] for l in V]
# n = len(V)
# V_ = brd_on_gsp(n, 3, V)
# print(V_)
# (P, M) = gsp_efficient(n, 3, V_)
# print(P, M)
# print(calc_utilities_efficient(V_real, P, M))


# print(brd_on_gsp(3, 2, [[0, 3], [0, 30], [0, 40]]))

def main():
    # TODO: Put your analysis and plotting code here, if any
    lec5_page7_example_q7b()
    lec5_page7_example_q8a()
    q8b_analysis()
    b2a_analysis()
    b2b_analysis()

if __name__ == "__main__":
    main()
