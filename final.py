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

def shortest_path_bi_dijkstra(file_path):
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

import heapq
import sys
import time

def euclidean_distance(x1, y1, x2, y2, k=2):
	"""Calculate the Euclidean distance between two points."""
	dx = x1 - x2
	dy = y1 - y2
	# special case (every edge has weight 1)

	if k == 0:
		return 1.0
	# Chebyshev distance    
	if k == -1:
		return max(abs(dx), abs(dy))
	#  k  >=  :  Minkowski distance
	return (abs(dx) ** k + abs(dy) ** k) ** (1.0 / k)

def a_star_search(graph, node_coords, start, goal,k):
	"""
	A* search algorithm to find shortest path between start and goal nodes.

	Args:
		graph: Dictionary representing the graph adjacency list
		node_coords: Dictionary of node coordinates {node_id: (x, y)}
		start: Starting node ID
		goal: Goal node ID
		
	Returns:
		path: List of nodes representing the shortest path, or None if no path exists
		total_cost: The cost of the path
		visited_count: Number of nodes visited during the search
	"""
	# Priority queue for nodes to explore (f_score, node_id)
	open_set = []

	# Initialize g_score (cost from start to current node)
	g_score = {node: float('inf') for node in graph}
	g_score[start] = 0

	# Initialize f_score (estimated total cost from start to goal through current node)
	f_score = {node: float('inf') for node in graph}
	f_score[start] = euclidean_distance(*node_coords[start], *node_coords[goal],k)

	# For reconstructing the path
	came_from = {}

	# Add start node to open set
	heapq.heappush(open_set, (f_score[start], start))

	# Set to keep track of nodes in the open set for faster lookup
	open_set_hash = {start}

	# Set to keep track of visited nodes
	visited_nodes = set()

	while open_set:
		# Get the node with lowest f_score
		current_f, current = heapq.heappop(open_set)
		open_set_hash.remove(current)
		
		# Mark current node as visited
		visited_nodes.add(current)
		
		# If we've reached the goal, reconstruct and return the path
		if current == goal:
			path = []
			total_cost = g_score[current]
			while current in came_from:
				path.append(current)
				current = came_from[current]
			path.append(start)
			path.reverse()
			return len(visited_nodes), total_cost,path
		
		# Explore neighbors
		for neighbor,weight in graph[current]:
			# Calculate tentative g_score
			# Cost is the Euclidean distance between current and neighbor
			cost = euclidean_distance(*node_coords[current], *node_coords[neighbor])
			tentative_g_score = g_score[current] + cost
			
			# If we found a better path to the neighbor
			if tentative_g_score < g_score[neighbor]:
				# Update the path
				came_from[neighbor] = current
				g_score[neighbor] = tentative_g_score
				f_score[neighbor] = tentative_g_score + euclidean_distance(*node_coords[neighbor], *node_coords[goal])
				
				# Add neighbor to open set if not already there
				if neighbor not in open_set_hash:
					heapq.heappush(open_set, (f_score[neighbor], neighbor))
					open_set_hash.add(neighbor)

	# No path found
	return len(visited_nodes),float('inf'), None

def read_graph_from_file(file_path):
	"""
	Read the graph from a file with the specified format.

	First line: n m k s t (number of nodes, number of edges, ?, start node, target node)
	Next n lines: x_i y_i (coordinates of node i)
	Next m lines: u_j v_j (edge between nodes u_j and v_j)

	Returns:
		n, m, k, s, t: Values from the first line
		node_coords: Dictionary of node coordinates {node_id: (x, y)}
		graph: Dictionary representing the graph adjacency list
	"""
	with open(file_path, 'r') as f:
		lines = f.readlines()

	# Parse first line
	n, m, k, s, t = map(int, lines[0].strip().split())

	# Parse node coordinates (1-indexed in input, convert to 0-indexed)
	node_coords = {}
	for i in range(1, n + 1):
		node_number, x, y = map(float, lines[i].strip().split())
		node_coords[node_number] = (x, y)

	# Build adjacency list
	graph = {i: [] for i in range(1, n + 1)}
	for i in range(n + 1, n + m + 1):
		u, v = map(int, lines[i].strip().split())
		graph[u].append(v)
		graph[v].append(u)  # Assuming undirected graph

	return n, m, k, s, t, node_coords, graph

def shortest_path_a_star():
	if len(sys.argv) != 2:
		print("Usage: python a_star_search.py <input_file_path>")
		return

	file_path = sys.argv[1]

	#n, m, k, s, t, nodes, graph = read_graph_from_file(file_path)
	n, m, k, s, t, nodes, edges = read_graph(file_path)
	graph = build_graph(nodes, edges, k)
	visited_count, total_cost, path= a_star_search(graph, nodes, s, t,k)
	
	if path:
		# Line 1: Number of visited nodes
		print(visited_count)
		
		# Line 2: Euclidean length of the shortest path (total cost)
		print(total_cost)
		
		# Line 3: The shortest path of nodes separated by spaces
		print(" ".join(map(str, path)))
	else:
		print(f"No path found from node {s} to node {t}")

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

def dijkstra(graph, s, t):
    """
    Run Dijkstra's algorithm to find the shortest path from s to t.
    
    :param graph: Adjacency list representation of the graph
    :param s: Source node
    :param t: Target node
    :return: Tuple of (visited_count, distances, previous)
    """
    import heapq
    
    # Initialize distances and visited nodes
    distances = {node_id: float('infinity') for node_id in graph}
    distances[s] = 0
    previous = {node_id: None for node_id in graph}
    visited_count = 0
    
    # Priority queue for Dijkstra's algorithm
    pq = [(0, s)]
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        visited_count += 1
        
        # If we've reached the target node, we're done
        if current_node == t:
            break
        
        # If we've found a longer path to the current node, skip it
        if current_distance > distances[current_node]:
            continue
        
        # Check all neighboring nodes
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            
            # If we've found a shorter path to the neighbor, update it
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))
    
    return visited_count, distances, previous

def reconstruct_path(previous, s, t):
    """
    Reconstruct the path from source to target.
    
    :param previous: Dictionary mapping each node to its predecessor
    :param s: Source node
    :param t: Target node
    :return: List of nodes in the path
    """
    path = []
    current = t
    
    while current is not None:
        path.append(current)
        current = previous[current]
    
    path.reverse()
    
    # Check if there is a valid path from s to t
    if not path or path[0] != s:
        return None
    
    return path

def shortest_path_dijkstra(graph_path: str):
    '''
    Find the shortest path between two nodes in a graph using Dijkstra's algorithm.
    
    :param graph_path: Path to the file containing the graph data
    '''
    # Read and parse the graph
    n, m, k, s, t, nodes, edges = read_graph(graph_path)
    
    # Build the graph
    graph = build_graph(nodes, edges, k)
    
    # Run Dijkstra's algorithm
    visited_count, distances, previous = dijkstra(graph, s, t)
    
    # Reconstruct the path
    path = reconstruct_path(previous, s, t)
    
    # Check if path exists
    if path is None:
        print("No path found from source to target")
        return None
    
    # Calculate the length of the shortest path
    path_length = distances[t]
    
    # Print the result
    print(visited_count)
    print(path_length)
    print(' '.join(map(str, path)))
    #print(len(path))
    
    return visited_count, path_length, path

def shortest_path(graph_path:str):
	n, m, k, s, t, nodes, edges = read_graph(graph_path)
	# Build the graph
	graph = build_graph(nodes, edges, k)
	import random
	if(n >= 5000 or m >= 500000):
		if (random.random() < 0.5):
			visited_count, total_cost, path = bi_dijkstra(graph, s, t)
		else:
			visited_count, total_cost, path= a_star_search(graph, nodes, s, t,k)
	else:
		if (random.random() < 0.5):
			visited_count, distances, previous = dijkstra(graph, s, t)
			path = reconstruct_path(previous, s, t)
			# Check if path exists
			if path is None:
				print("No path found from source to target")
				return None
			# Calculate the length of the shortest path
			total_cost = distances[t]
		else:
			visited_count, total_cost, path= a_star_search(graph, nodes, s, t,k)
	# Print the result
	print(visited_count)
	print(total_cost)
	print(' '.join(map(str, path)))
	return visited_count, total_cost, path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        import time
        start_time = time.time()
        shortest_path(sys.argv[1])
        print("--- %s seconds ---" % (time.time() - start_time))