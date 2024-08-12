# Skeleton file for HW3 questions 9 and 10
# =====================================
# IMPORTANT: You are NOT allowed to modify the method signatures
# (i.e. the arguments and return types each function takes).
# We will pass your grade through an autograder which expects a specific format.
# =====================================


# Do not include any other files or an external package, unless it is one of
# [numpy, pandas, scipy, matplotlib, random]
# please contact us before sumission if you want another package approved.
import numpy as np
import matplotlib.pyplot as plt
from hw3_matchingmarket import market_eq


def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


# === Problem 9(a) ===
def exchange_network_from_uber(
    n, m, l, rider_vals, rider_locs, rider_dests, driver_locs
):
    """Given a market scenario for ridesharing, with n riders, and
    m drivers, output an exchange network representing the problem.
    -   The grid is size l x l. Points on the grid are tuples (x,y) where
        both x and y are in (0...l).
    -   rider_vals is a list of numbers, where rider_vals[i] is the value
        of rider i's trip
    -   rider_locs is a list of points, where rider_locs[i] is the current
        location of rider i (in 0...n-1)
    -   rider_dests is a list of points, where rider_dests[i] is the desired
        destination of rider i (in 0...n-1)
    -   driver_locs is a list of points, where driver_locs[j] is the current
        location of driver j (in 0...m-1)
    Output a tuple (n, m, V) representing a bipartite exchange network, where:
    -   V is an n x m list, with V[i][j] is the value of the edge between
        rider i (in 0...n-1) and driver j (in 0...m-1)"""
    V = np.zeros((n, m))
    # cost of i,j is the distance between rider i and driver j + distance between rider i and rider i's destination
    # value is the value of the rider minus cost
    for i in range(n):
        rider_dest_dist = manhattan_distance(rider_locs[i], rider_dests[i])
        for j in range(m):
            cost = manhattan_distance(rider_locs[i], driver_locs[j]) + rider_dest_dist
            V[i][j] = max(rider_vals[i] - cost, 0)
    return (n, m, V)


# === Problem 10 ===
def stable_outcome(n, m, V):
    """Given a bipartite exchange network, with n riders, m drivers, and
    edge values V, output a stable outcome (M, A_riders, A_drivers).
    -   V is defined as in exchange_network_from_uber.
    -   M is an n-element list, where M[i] = j if rider i is
        matched with driver j, and M[i] = None if there is no matching.
    -   A_riders is an n-element list, where A_riders[i] is the value
        allocated to rider i.
    -   A_drivers is an m-element list, where A_drivers[j] is the value
        allocated to driver j."""
    A_riders = [0] * n
    A_drivers = [0] * m
    P, M = market_eq(n, m, V)
    for i in range(n):
        matched_driver = M[i]
        if matched_driver is None:
            continue
        A_riders[i] = V[i][matched_driver] - P[matched_driver]
        A_drivers[matched_driver] = P[matched_driver]
    return (M, A_riders, A_drivers)


# === Problem 10(a) ===
def rider_driver_example_1():
    n = 5
    m = 5
    l = 20
    rider_vals = [30] * 5
    rider_locs = [(2, 10), (6, 10), (10, 10), (14, 10), (18, 10)]
    rider_dests = [(2, 15), (6, 15), (10, 15), (14, 15), (18, 15)]
    driver_locs = [(2, 2), (6, 6), (10, 10), (14, 14), (18, 18)]
    return (n, m, l, rider_vals, rider_locs, rider_dests, driver_locs)

def rider_driver_example_2():
    n = 10
    m = 5
    l = 20
    rider_vals = [50, 45, 40, 35, 30, 25, 30, 35, 40, 45]
    rider_locs = [(1, 10), (3, 10), (5, 10), (7, 10), (9, 10), (11, 10), (13, 10), (15, 10), (17, 10), (19, 10)]
    rider_dests = [(1, 15), (3, 15), (5, 15), (7, 15), (9, 15), (11, 15), (13, 15), (15, 15), (17, 15), (19, 15)]
    driver_locs = [(2, 2), (6, 6), (10, 10), (14, 14), (18, 18)]
    return (n, m, l, rider_vals, rider_locs, rider_dests, driver_locs)

def q10a_analysis():

    # analyze the stable outcomes for the two examples
    def analyze_stable_outcome(example, example_tag=""):
        # get the stable outcome for the example
        stable_outcome_example = stable_outcome(*exchange_network_from_uber(*example))

        # get the values from the example
        n, m, l, rider_vals, rider_locs, rider_dests, driver_locs = example
        M, A_riders, A_drivers = stable_outcome_example

        # print the total value allocated to riders and drivers for the example
        total_value = sum(A_riders) + sum(A_drivers)
        print(f"""
        Example {example_tag}:
            Total value allocated to riders and drivers: {total_value}
            Matching: {M=}
            Driver Allocations: {A_drivers}
            Rider Allocations: {A_riders}
            Driver Profits: {A_drivers}
            Rider Prices: {list(np.array(rider_vals) - np.array(A_riders))}
        """)

        # plot the riders and drivers
        for i in range(n):
            if M[i] is not None:
                plt.plot([rider_locs[i][0], rider_dests[i][0]], [rider_locs[i][1], rider_dests[i][1]], 'r')
                plt.plot([driver_locs[M[i]][0], rider_locs[i][0]], [driver_locs[M[i]][1], rider_locs[i][1]], 'b')

        # plot the grid
        plt.xlim(0, l)
        plt.ylim(0, l)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(f"q10a_{n=}_{m=}_{example_tag=}.pgf", format="pgf")
        plt.savefig(f"q10a_{n=}_{m=}_{example_tag=}.png", format="png")
        plt.show()

    analyze_stable_outcome(rider_driver_example_1(), "1")
    analyze_stable_outcome(rider_driver_example_2(), "2")

# === Problem 10(b) ===
def random_riders_drivers_stable_outcomes(n, m):
    """Generates n riders, m drivers, each located randomly on the grid,
    with random destinations, each rider with a ride value of 100,
    and returns the stable outcome."""
    value = 100

    # generate random riders, drivers, and destinations
    rider_vals = [value] * n
    rider_locs = np.random.randint(0, 100, (n, 2))
    rider_dests = np.random.randint(0, 100, (n, 2))
    driver_locs = np.random.randint(0, 100, (m, 2))

    # generate exchange network
    n, m, V = exchange_network_from_uber(n, m, 100, rider_vals, rider_locs, rider_dests, driver_locs)

    # get stable outcome
    return stable_outcome(n, m, V)

def random_riders_drivers_stable_outcomes_for_analysis(n, m):
    """Generates n riders, m drivers, each located randomly on the grid,
    with random destinations, each rider with a ride value of 100,
    and returns the stable outcome."""
    value = 100

    # generate random riders, drivers, and destinations
    rider_vals = [value] * n
    rider_locs = np.random.randint(0, 100, (n, 2))
    rider_dests = np.random.randint(0, 100, (n, 2))
    driver_locs = np.random.randint(0, 100, (m, 2))

    # generate exchange network
    n, m, V = exchange_network_from_uber(n, m, 100, rider_vals, rider_locs, rider_dests, driver_locs)

    # get stable outcome
    return stable_outcome(n, m, V), rider_vals

def q10b_analysis():

    # given an amount of riders and drivers, generate 100 random stable outcomes and analyze them
    def analyze_outcomes(n, m):
        # get 100 random stable outcomes
        results = [random_riders_drivers_stable_outcomes_for_analysis(n, m) for _ in range(100)]
        
        # get the list of profits for the drivers
        driver_profits = [results[i][0][2] for i in range(100)]

        # driver match amount
        driver_match_amount = [sum([1 if j in results[i][0][0] else 0 for i in range(100)]) for j in range(m)]

        # get the list of estimated prices for the riders
        rider_prices = [list(np.array(results[i][1]) - np.array(results[i][0][1])) for i in range(100)]

        # rider match amount
        rider_match_amount = [sum([1 if results[i][0][0][j] is not None else 0 for i in range(100)]) for j in range(n)]

        # plot profits histogram for each driver
        for i in range(m):

            # profits list
            driver_profits_list = [driver_profits[j][i] for j in range(100)]

            # print avg, min, max, median, and std
            avg = np.mean(driver_profits_list)
            min = np.min(driver_profits_list)
            max = np.max(driver_profits_list)
            median = np.median(driver_profits_list)
            std = np.std(driver_profits_list)
            print(f"Driver {i} Profits for {n=}, {m=}: {avg=}, {min=}, {max=}, {median=}, {std=}, {driver_match_amount[i]=}")

            # plot histogram and the min, max, median (std in legend)
            plt.figure() # create a new figure
            plt.hist(driver_profits_list, bins=5)
            plt.axvline(avg, color='k', linestyle='--', linewidth=1, label=f"Average={avg:.2f}")
            plt.axvline(min, color='r', linestyle='--', linewidth=1, label=f"Min={min:.2f}")
            plt.axvline(max, color='y', linestyle='--', linewidth=1, label=f"Max={max:.2f}")
            plt.axvline(median, color='g', linestyle='--', linewidth=1, label=f"Median={median:.2f}")
            plt.legend(title=f'Standard Deviation: {std:.2f}\nMatch Amount: {driver_match_amount[i]}')
            plt.title(f"Driver Profits for {n=}, {m=}")
            plt.xlabel("Profits")
            plt.ylabel("Frequency")
            plt.savefig(f"q10b_{n=}_{m=}_driver_{i}_profits.png")
            plt.savefig(f"q10b_{n=}_{m=}_driver_{i}_profits.pgf")
            
        # plot prices histogram for each rider
        for i in range(n):
            
            # prices list
            rider_prices_list = [rider_prices[j][i] for j in range(100)]
            
            # print avg, min, max, median, and std
            avg = np.mean(rider_prices_list)
            min = np.min(rider_prices_list)
            max = np.max(rider_prices_list)
            median = np.median(rider_prices_list)
            std = np.std(rider_prices_list)
            print(f"Rider {i} Prices for {n=}, {m=}: {avg=}, {min=}, {max=}, {median=}, {std=}, {rider_match_amount[i]=}")

            # plot
            plt.figure() # create a new figure
            plt.hist(rider_prices_list, bins=5)
            plt.axvline(avg, color='k', linestyle='--', linewidth=1, label=f"Average={avg:.2f}")
            plt.axvline(min, color='r', linestyle='--', linewidth=1, label=f"Min={min:.2f}")
            plt.axvline(max, color='y', linestyle='--', linewidth=1, label=f"Max={max:.2f}")
            plt.axvline(median, color='g', linestyle='--', linewidth=1, label=f"Median={median:.2f}")
            plt.legend(title=f'Standard Deviation: {std:.2f}\nMatch Amount: {rider_match_amount[i]}')
            plt.title(f"Rider Prices for {n=}, {m=}")
            plt.xlabel("Prices")
            plt.ylabel("Frequency")
            plt.savefig(f"q10b_{n=}_{m=}_rider_{i}_prices.png")
            plt.savefig(f"q10b_{n=}_{m=}_rider_{i}_prices.pgf")

    # n = m = 10
    analyze_outcomes(10, 10)

    # 5 = n < m = 20
    analyze_outcomes(5, 20)
    
    # 20 = n > m = 5
    analyze_outcomes(20, 5)

# === Bonus 3(a) (Optional) ===
def public_transport_stable_outcome(
    n, m, l, rider_vals, rider_locs, rider_dests, driver_locs, a, b
):
    """Given an l x l grid, n riders, m drivers, and public transportation
    parameters (a,b), output a stable outcome (M, A_riders, A_drivers), where:
    -   rider_vals, rider_locs, rider_dests, driver_locs are defined the same
        way as in exchange_network_from_uber
    -   the cost of public transport is a + b * dist(start, end) where dist is
        manhattan distance
    -   M is an n-element list, where M[i] = j if rider i is
        matched with driver j, and M[i] = -1 if rider i takes public transportation, and M[i] = None if there is no match for rider i.
    -   A_riders, A_drivers are defined as before.
    -   If there is no stable outcome, return None.
    """
    # build the exchange network
    A_riders = [0] * n
    A_drivers = [0] * m
    V = np.zeros((n, m + n))
    for i in range(n):
        rider_dest_dist = manhattan_distance(rider_locs[i], rider_dests[i])
        public_transport_price = a + b * rider_dest_dist
        for j in range(m):
            cost = manhattan_distance(rider_locs[i], driver_locs[j]) + rider_dest_dist
            V[i][j] = max(rider_vals[i] - cost, 0)
        for j in range(m, m + n):
            V[i][j] = max(rider_vals[i] - public_transport_price, 0)

    # get the stable outcome
    P, M = market_eq(n, m + n, V)
    for i in range(n):
        matched_driver = M[i]
        if matched_driver is None:
            continue
        elif matched_driver >= m:
            A_riders[i] = V[i][matched_driver]
            M[i] = -1
        else:
            A_riders[i] = V[i][matched_driver] - P[matched_driver]
            A_drivers[matched_driver] = P[matched_driver]

    # return the stable outcome
    return (M, A_riders, A_drivers)


def b3a_analysis():
    # for a lot of random riders and drivers, run the public_transport_stable_outcome function
    for i in range(100):
        n = np.random.randint(5, 20)
        m = np.random.randint(5, 20)
        l = 100
        rider_vals = [np.random.randint(20, 100) for _ in range(n)]
        rider_locs = np.random.randint(0, l, (n, 2))
        rider_dests = np.random.randint(0, l, (n, 2))
        driver_locs = np.random.randint(0, l, (m, 2))
        a = np.random.randint(5, 20)
        b = np.random.randint(1, 5)
        result = public_transport_stable_outcome(n, m, l, rider_vals, rider_locs, rider_dests, driver_locs, a, b)
        print(f"Test {i}: {result=}")

def main():
    # TODO: Put your analysis and plotting code here, if any
    q10a_analysis()
    q10b_analysis()
    b3a_analysis()


if __name__ == "__main__":
    main()
