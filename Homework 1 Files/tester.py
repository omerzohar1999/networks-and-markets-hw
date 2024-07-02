import hw1

# DO NOT SUBMIT THIS FILE
# It is just an example of a few tests that we will run on your code that you can use as a starting point
# to make sure the code is correct.
# You should put the two python files in the same folder and run this one

testGraph = hw1.UndirectedGraph(5)

assert testGraph.number_of_nodes() == 5

assert testGraph.check_edge(0,1) == False

testGraph.add_edge(0,1)

assert testGraph.check_edge(0,1) == True
assert testGraph.check_edge(1,0) == True

testGraph.add_edge(2,1)

assert testGraph.check_edge(1,2) == True

testGraph2 = hw1.create_graph(100, 0.5)
assert testGraph2.number_of_nodes() == 100
outboundNode1List = testGraph2.edges_from(1)
assert len(outboundNode1List) > 2  # With very, very small probability this test will fail even for correct implementation

assert hw1.shortest_path(testGraph, 0, 2) == 2
assert hw1.shortest_path(testGraph, 1, 2) == 1
assert hw1.shortest_path(testGraph, 1, 3) == -1

testGraph3 = hw1.UndirectedGraph(3)
assert testGraph3.number_of_nodes() == 3
testGraph3.add_edge(0,1)
testGraph3.add_edge(2,1)
assert hw1.avg_shortest_path(testGraph3) > 1.2 # Should be close to true value of 1.3333
assert hw1.avg_shortest_path(testGraph3) < 1.5 # Should be close to true value of 1.3333

# Now, assuming that facebook_combined.txt is in the same directory as tester.py
testFbGraph = hw1.create_fb_graph("facebook_combined.txt")
assert testFbGraph.number_of_nodes() == 4039
assert testFbGraph.check_edge(107, 1453) == True
assert testFbGraph.check_edge(133, 800) == False

print("all tests passed")
