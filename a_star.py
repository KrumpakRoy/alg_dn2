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
		for neighbor in graph[current]:
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
        #distance = euclidean_distance(nodes[u][0], nodes[u][1],nodes[v][0], nodes[v][1], k)
        
        # Add edges in both directions as the graph is undirected
        graph[u].append(v)
        graph[v].append(u)
    
    return graph

def main():
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
			

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: python a_star_search.py <input_file_path>")
		sys.exit(1)
	time_start = time.time()
	main()
	print(f"Elapsed time: {time.time() - time_start:.3f} seconds")