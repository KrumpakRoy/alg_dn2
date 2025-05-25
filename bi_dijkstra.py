import heapq
from collections import defaultdict, deque
import queue
#from generate_graph import visualize_graph
def calculate_distance(u_coords, v_coords, k):
	"""
	Calculate the distance between two nodes based on the k value.

	:param u_coords: Coordinates of node u (x, y)
	:param v_coords: Coordinates of node v (x, y)
	:param k: Value of k for the distance metric
	:return: Distance between nodes u and v
	"""
	xu, yu = u_coords
	xv, yv = v_coords

	if k == 0:
		return 1
	elif k == -1:
		return max(abs(xu - xv), abs(yu - yv))
	else:
		return (abs(xu - xv) ** k + abs(yu - yv) ** k) ** (1 / k)

def get_path(previous_forward, previous_backward, meeting_node):
	path_forward = []
	current = meeting_node
	#print(previous_backward)
	#print(previous_forward)
	while current is not None:
		path_forward.append(current)
		current = previous_forward.get(current)
	path_forward.reverse()
	
	path_backward = []
	current = previous_backward.get(meeting_node)
	while current is not None:
		path_backward.append(current)
		current = previous_backward.get(current)
	
	full_path = path_forward + path_backward
	
	return full_path

def bi_dijkstra(graph, s, t):
	#initialize best shortest paths to nodes
	#from bakcward and forward direction
	distances_forward = defaultdict(lambda: float("inf"))
	distances_forward[s] = 0
	distances_backward = defaultdict(lambda: float("inf"))
	distances_backward[t] = 0

	#initialize priority queues
	forward_pq = queue.PriorityQueue()
	forward_pq.put((0,s))
	backward_pq = queue.PriorityQueue()
	backward_pq.put((0, t))

	best_path = float("inf")
	seen_nodes_forward = set()
	seen_nodes_backward = set()

	previous_nodes_forward = defaultdict(lambda: None)
	previous_nodes_backward = defaultdict(lambda: None)
	best_meeting_point = None
	visited = 0
	while (not forward_pq.empty()) and (not backward_pq.empty()):
		u = forward_pq.get()[1]  # get is a "pop"
		v = backward_pq.get()[1]
		visited += 2
		seen_nodes_forward.add(u)
		seen_nodes_backward.add(v)

		for x,distance_x_u in graph[u]: 
			# check if you found a shorter path to the node x
			if (x not in seen_nodes_forward) and distances_forward[x] > distances_forward[u] + distance_x_u:
				distances_forward[x] = distances_forward[u] + distance_x_u
				previous_nodes_forward[x] = u
				forward_pq.put((distances_forward[x], x))
			# check if the forward path to u and the backward path to x sum with distance from u to x into a better path 
			#than the current best path length found
			if (x in seen_nodes_backward) and (distances_forward[u] + distance_x_u + distances_backward[x] < best_path):
				best_path = distances_forward[u] + distance_x_u + distances_backward[x]
				best_meeting_point = x

		for x,distance_x_v in graph[v]:
			# check if you found a shorter path to the node x
			if (x not in seen_nodes_backward) and distances_backward[x] > distances_backward[v] + distance_x_v:
				distances_backward[x] = distances_backward[v] + distance_x_v
				previous_nodes_backward[x] = v
				backward_pq.put((distances_backward[x], x))
			# check if the forward path to x and the backward path to v sum with distance from v to x into a better path 
			#than the current best path length found
			if (x in seen_nodes_forward) and (distances_backward[v] + distance_x_v + distances_forward[x] < best_path):
				best_path = distances_backward[v] + distance_x_v + distances_forward[x]
				best_meeting_point = x
		#once the paths are longer, you found the best path 
		if distances_forward[u] + distances_backward[v] >= best_path:
			path = get_path(previous_nodes_forward, previous_nodes_backward, best_meeting_point)
			return visited, best_path, path
	#print("There was an error (one queue is empty while the other is not). This might happen if the graph is not connected? ... or maybe some other reason, it would be best to first check if the graph is connected.")
	return visited, best_path, get_path(previous_nodes_forward, previous_nodes_backward, best_meeting_point)

def read_graph(graph_path):
    """
    Read and parse graph data from the file.
    
    :param graph_path: Path to the file containing the graph data
    :return: Tuple of (n, m, k, s, t, nodes, edges)
    """
    with open(graph_path, 'r') as file:
        lines = file.readlines()
    
    # Parse the first line to get n, m, k, s, t
    n, m, k, s, t = map(int, lines[0].split())
    
    # Parse the nodes
    nodes = {}
    for i in range(1, n + 1):
        node_id, x, y = map(float, lines[i].split())
        nodes[node_id] = (x, y)
    
    # Parse the edges
    edges = []
    for i in range(n + 1, n + m + 1):
        u, v = map(int, lines[i].split())
        edges.append((u, v))
    
    return n, m, k, s, t, nodes, edges

def build_graph(nodes, edges, k):
    """
    Build the graph as an adjacency list with calculated distances.
    
    :param nodes: Dictionary of nodes with their coordinates
    :param edges: List of edges as (u, v) pairs
    :param k: Value of k for the distance metric
    :return: Adjacency list representation of the graph
    """
    graph = {node_id: [] for node_id in nodes}
    
    for u, v in edges:
        # Calculate the distance based on the value of k
        distance = calculate_distance(nodes[u], nodes[v], k)
        
        # Add edges in both directions as the graph is undirected
        graph[u].append((v, distance))
        graph[v].append((u, distance))
    
    return graph

def shortest_path(file_path):
    """# Read and parse the graph file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Parse the first line
    n, m, k, s, t = map(int, lines[0].strip().split())
    # Parse node coordinates
    nodes = {}
    for i in range(1, n + 1):
        node_id, x, y = map(int, lines[i].strip().split())
        nodes[node_id] = (x, y)
    # Build adjacency list with edge weights (Minkowski distance)
    graph = defaultdict(list)
    edges = []
    for i in range(n + 1, n + 1 + m):
        u, v = map(int, lines[i].strip().split())
        # Calculate Minkowski distance
        distance = calculate_distance(nodes[u], nodes[v],k)
        graph[u].append((v, distance))
        graph[v].append((u, distance))
        edges.append((u,v))
    # Run bidirectional Dijkstra"""
    n, m, k, s, t, nodes, edges = read_graph(file_path)
    graph = build_graph(nodes, edges, k)
    #visualize_graph(nodes.values(), edges)
    visited_count, path_length, path = bi_dijkstra(graph, s, t)
    # Output results
    print(visited_count)
    print(f"{path_length:.11f}")
    print(' '.join(map(str, path)))
    return visited_count, path_length, path
if __name__ == "__main__":
	import sys
	if len(sys.argv) > 1:
		import time
		start_time = time.time()
		shortest_path(sys.argv[1])
		print("--- %s seconds ---" % (time.time() - start_time))
# Example usage:
# shortest_path('graph.txt')