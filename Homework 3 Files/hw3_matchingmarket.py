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
    for origin in range(G.number_of_nodes()):
        for destination in G.edges_from(origin):
            capacity = G.get_edge(origin, destination)
            flow = F.get_edge(origin, destination)
            if flow < capacity:
                R.set_edge(origin, destination, capacity - flow)
            if flow > 0:
                R.set_edge(destination, origin, flow)
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
            F.set_edge(origin, destination, F.get_edge(origin, destination) + flow)
        residual = residual_graph(G, F)
        path = residual.get_path(s, t)

    # calculate v
    v = 0
    for neighbor in F.edges_from(s):
        v += F.get_edge(s, neighbor)
    return v, F


def max_matching(n, C):
    """Given integer n, and a set of matching constraints C,
    output a maximum matching. Specifically, C is a n x m array, where
    C[i][j] = 1 if left i (in 0...n-1) and right j (in 0...m-1) are compatible.
    If left i and right j are incompatible, then C[i][j] = 0.
    Return an n-element array M where M[i] = j if left i is matched with right j,
    and M[i] = None if left i is not matched."""
    pass

def max_weight_matching(C: 'n x m array') -> 'players assignment':
    """return a maximum weight matching, C[i, j] is the weight of the edge"""
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
    match_graph = WeightedDirectedGraph(2 * (n + 1))
    source = 0
    sink = 2 * n + 1
    for i in range(n):
        match_graph.set_edge(source, i + 1, 1)
        for j in filter(lambda j: C[i][j], range(n)):
            match_graph.set_edge(i + 1, n + j + 1, 1)
    for j in range(n):
        match_graph.set_edge(n + j + 1, sink, 1)
    flow_val, F = max_flow(match_graph, source, sink)
    if flow_val < n:
        residual = residual_graph(match_graph, F)
        residual.single_source_bfs(source)
        return False, [i for i in range(n) if residual.distances[source][i + 1] != -1]

    matching = []
    for i in range(n):
        for j in range(n):
            if F.get_edge(i + 1, n + j + 1) == 1:
                matching.append(j)
                break
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
        for j in range(m):
            C[i][j] = 1 if V[i][j] - P[j] == maximal else 0
    return C


def make_square_prices(V):
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
    V_square = make_square_prices(V)
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
    return P[:m], M[:n]


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


# === Problem 8(a) ===
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
    """TODO: I assume non-negative valuations!!"""
    """TODO: needs work on when n!=m"""
    rect_V = rectangular_valuations(V)
    # matching
    M = max_weight_matching(rect_V)[:n]
    # payments
    sum_value = sum(V[i][M[i]] for i in range(n))
    P = np.array([sum_value - V[i][M[i]] for i in range(n)])
    # pivot
    H = np.zeros((n,))
    for i in range(n):
        row_i = copy(rect_V[i])
        rect_V[i] = [0] * m
        i_M = max_weight_matching(rect_V)
        H[i] = sum(rect_V[i][i_M[i]] for i in range(n))
        rect_V[i] = row_i

    # item (-cost) i.e. positive
    C = np.zeros((m,))
    for i in range(n):
        j = M[i]
        C[i] += H[j] - P[j]

    return list(C), list(M)


# === Bonus Question 2(a) (Optional) ===
def random_bundles_valuations(n, m):
    """Given n buyers, m bundles, generate a matching market context
    (n, m, V) where V[i][j] is buyer i's valuation for bundle j.
    Each bundle j (in 0...m-1) is comprised of j copies of an identical good.
    Each player i has its own value for an individual good; this value is sampled
    uniformly at random from [1, 50] inclusive, for each player"""

    costs = np.random.randint(0, 51, size=n)
    V = np.tile(np.arange(0, m), (n, 1))
    return (V.T * costs).T

# === Bonus Question 2(b) (optional) ===

def gsp_efficient(n, m, V) -> '(P, M)':
    """ here V is the valuation of a single item"""
    M = np.argsort(V)
    P = np.zeros((m,), dtype=int)

    for i in range(min(m, n-1)):
        P[-(i+1)] = V[M[-(i+2)]] * (m-1-i)

    return list(P), [i if i >= 0 else None for i in (M+m-n)]


def calc_utilities_efficient(V, P, M):
    return [(j * v - P[j])
            if j is not None else 0
            for j, v in zip(M, V)]

# (P, M) = gsp_efficient(3, 3, [3, 41, 40])
# print(P, M)
# print(calc_utilities_efficient([3, 40, 40], M, P))

def gsp(n, m, V) -> '(P, M)':
    """Given a matching market for bundles with n buyers, and m bundles, and
    valuations V (for bundles), output a tuple (P, M) of prices P and a
    matching M as computed using GSP."""
    V = [V[i][1] for i in range(n)]  # valuation of a single item.
    return gsp_efficient(n, m, V)

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
    print("hello world")


if __name__ == "__main__":
    main()
