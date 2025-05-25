import random
#from fibonacci_heap import FibonacciHeap
import numpy as np
from generate_graph import visualize_graph
import heapq
from hollow_heap import HollowHeap
from heapdict import heapdict
def read_graph(graph_path):
	"""
	Read and parse graph data from the file.
	:param graph_path: Path to the file containing the graph data
	(s needs to be in it)
	:return: Tuple of (n, m, k, s, t, nodes, edges, R)
	"""
	with open(graph_path, 'r') as file:
		lines = file.readlines()

	# Parse the first line to get n, m, k, s, t
	n, m, k, s, t = map(int, lines[0].split())
	probability = 1/np.sqrt(np.log10(n)/np.log10(np.log10(n)))
	R = {}
	# Parse the nodes
	nodes = {}
	for i in range(1, n + 1):
		node_id, x, y = map(float, lines[i].split())
		if random.random() < probability or i == s:
			R[node_id] = (x, y)
		nodes[node_id] = (x, y)
	#while(len(R) < n * probability):
	#	#print(f"len R < n*probability, {len(R)} < {n} * {probability}")
	#	i = random.randint(1,n)
	#	while( i in R.keys()):
	#		i = random.randint(1,n)
	#	R[i] = nodes[i]
	# Parse the edges
	edges = []
	for i in range(n + 1, n + m + 1):
		u, v = map(int, lines[i].split())
		edges.append((u, v))

	return n, m, k, s, t, nodes, edges, R

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
		#distance = calculate_distance(nodes[u], nodes[v], k)
		
		# Add edges in both directions as the graph is undirected
		#graph[u].append((v, distance))
		graph[u].append(v)
		#graph[v].append((u, distance))
		graph[v].append(u)

	return graph

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

def dijkstra_for_V_without_R(graph, v, R, nodes,k):
	"""
	Run Dijkstra's algorithm to find the shortest path from 
	each vertex in graph to a node in R
	:param graph: Adjacency list representation of the graph
	:param v: the vertex for which we calculate b(v) = r from R, 
	which first exists the heap when searching for the shortest
		path from v to s
	:param R: Subset of vertices of the graph
	:return: Tuple of (visited_count, distances, previous)
	"""
	import heapq

	# Initialize distances and visited nodes
	distances = {node_id: float('infinity') for node_id in graph}
	distances[v] = 0
	previous = {node_id: None for node_id in graph}
	ball = []
	# Priority queue for Dijkstra's algorithm
	pq = [(0, v)]

	while pq:
		current_distance, current_node = heapq.heappop(pq)

		# If we've reached the target node, we're done
		if current_node in R:
			return current_distance, current_node, ball
		ball.append((current_node, current_distance))
		# Check all neighboring nodes
		for neighbor in graph[current_node]:
			weight = calculate_distance(nodes[current_node], nodes[neighbor], k)
			distance = current_distance + weight
			# If we've found a shorter path to the neighbor, update it
			if distance < distances[neighbor]:
				distances[neighbor] = distance
				previous[neighbor] = current_node
				heapq.heappush(pq, (distance, neighbor))

	#return visited_count, distances, previous
def bundleConstrucion(graph,s,probability, k):
	R1 = [s]
	for node in graph:
		if random.random() < probability:
			R1.append(node)
	R2 = []
	b = {node_id: node_id for node_id in graph}
	Vs = {v:[] for v in graph if v not in R1}
	for v in graph:
		if (v not in R1):
			#heap = FibonacciHeap()
			heap = HollowHeap()
			#heap.insert(0, v)
			heap.push(0, v)
			V = Vs[v]
			#while heap.total_nodes > 0:
			while not heap.empty():
				#u = heap.extract_min()
				u = heap.find_min()
				heap.delete_min()
				#u_node = u.value()
				u_node = u.item
				u_key = u.key()
				V.append(u)
				if u_node in R1:
					break
				elif len(V) > probability * np.log(probability):
					R2.append(v)
					break
				else:
					for (x, weight) in graph[u_node]:
						heap_node_x = heap.in_heap(x)
						if heap_node_x != None and x not in V:
							#heap.insert(u_key+weight,x)
							heap.push(u_key+weight, x)
						elif heap_node_x.key > u_key + weight:
							heap.decrease_key(heap_node_x, u_key+weight)
			Vs[v] = V
	R = R1.extend(R2)
	for v in graph:
		if v not in R:
			i = 0
			V = Vs[v]
			while i < len(V):
				b_node = V[i]
				#b_node_id = b_node.value
				b_node_id = b_node.item
				b_node_key = b_node.key
				if b_node_id in R:
					b[v] = b_node_id
					i = len(V)
				i+=1
	#TODO
	bundles = {node: [] for node in R}
	balls = {node: [] for node in graph if node not in R}
	for v in Vs:
		bundles[Vs[v][-1]].append(v) 
		balls[v].extend(Vs[v][:-1])
	



def construct_bundles(graph, R, nodes,k):
	bundles = {node_id: [] for node_id in R}
	b = {}
	balls = {}
	#print("constructing bundles")
	for node_id in graph:
		#print(node_id)
		if (node_id) in R:
			b[node_id] = node_id
			#balls[node_id] = [(int(node_id), 0)]
			bundles[node_id].append((node_id, 0))
			continue
		else:
			distance, node_in_R, ball = dijkstra_for_V_without_R(graph, node_id, R, nodes,k)
			bundles[node_in_R].append((node_id,distance))
			balls[node_id] = ball
			b[node_id] = node_in_R
	return bundles, balls, b

def bundle_dijkstra(graph,s,R,k, nodes,edges):
	def relax(v,D):
		b_v = b[v]
		if D < distances[v]:
			distances[v] = D
			heap_node = in_heap[v]
			if heap_node != None:
				#heap.decrease_key(heap_node,D)
				hd[D] = heap_node
			elif v not in R.keys():
				relax(b_v, distances[v]+calculate_distance( nodes[v], nodes[b_v],k))
	#print(s)
	#print(R)
	bundles, balls, b = construct_bundles(graph, R, nodes,k)
	#print("bundles:\n",bundles)
	#print("balls:\n",balls)
	#print("b:\n",b)
	#visualize_graph(nodes.values(), edges, R=R)
	#heap = FibonacciHeap()
	#heap = HollowHeap()
	hd = heapdict()
	in_heap = {node_id: None for node_id in graph }
	distances = {node_id: float('infinity') for node_id in graph}
	distances[s] = 0
	for node_id in R:
		#n = heap.insert(distances[node_id],node_id)
		#n = heap.push(distances[node_id], node_id)
		#in_heap[node_id] = node_id
		hd[distances[node_id]] = node_id
		in_heap[node_id] = node_id
	#print(heap, heap.nodes_used)
	#print(in_heap)
	#s_node = heap.insert(0, s)
	#in_heap[s] = s_node
	visited_count = 0
	#print("==================")
	#while heap.total_nodes > 0:
	#while not heap.empty():
	while hd:
		#print(distances)
		#u = heap.extract_min()
		#u = heap.find_min()
		#node_u = int(u.value)
		#node_u = int(u.item)
		#distance_u = u.key
		distance_u, node_u = hd.popitem()
		#print(node_u, distance_u)
		visited_count += 1
		in_heap[node_u] = None
		for (v,distance_v) in bundles[node_u]:
			#print("relaxing")
			#relax 
			if (v != node_u):
				#print(f"v = {v}, node = {nodes[v]}")
				#print("relaxing v,d(u)+d(u,v)", f"v = {nodes[v]}, D = {distances[node_u]+calculate_distance(nodes[node_u],nodes[v],k)}")
				#print("relaxing v,d(u)+d(u,v)", f"v = {nodes[v]}, D = {distances[node_u]+distance_v}")
				#relax(v, distances[node_u]+calculate_distance(nodes[node_u],nodes[v],k))
				relax(v, distances[node_u]+distance_v)
				for (y, distance_y) in balls[v]:
					#if(y != v):
						#print(f"y = {y}, node = {nodes[y]}")
						#print("relaxing v,d(y)+d(y,v)", f"v = {nodes[v]}, D = {distances[y]} + {calculate_distance(nodes[y],nodes[v],k)} ={distances[y] + calculate_distance(nodes[y],nodes[v],k) }")
						#print("relaxing v,d(y)+d(y,v)", f"v = {nodes[v]}, D = {distances[y]} + {distance_y} ={distances[y] + distance_y}")
						#relax(v, distances[y] + calculate_distance(nodes[y],nodes[v],k))
						relax(v, distances[y] + distance_y)
				ball_v = balls[v]
				#ball_v.append((v, 0))
				for (z2, distance_z2) in ball_v:
					for z1 in graph[z2]:
						#if (z2 != v):
							#print(f"z2 = {z2}, node = {nodes[z2]}")
							#print(f"z1 = {z1}, node = {nodes[z1]}")
							#print("relaxing v,d(z1)+w11+d(z2,v)", f"v = {nodes[v]}, D = {distances[z1] + distance_z1 + calculate_distance(nodes[z2],nodes[v],k)}")
							#relax(v,distances[z1] + distance_z1 + calculate_distance(nodes[z2],nodes[v],k))
							distance_z1 = calculate_distance(nodes[z1], nodes[z2],k)
							#print("relaxing v,d(z1)+w11+d(z2,v)", f"v = {nodes[v]}, D = {distances[z1] + distance_z1 + distance_z2}")
							relax(v,distances[z1] + distance_z1 + distance_z2)
			for (x, distance_x) in bundles[node_u]:
				for y in graph[x]:
					#print(f"x = {x}, node = {nodes[x]}")
					#print(f"y = {y}, node = {nodes[y]}")
					distance_y = calculate_distance(nodes[x], nodes[y], k)
					#print("relaxing y,d(x)+wxy", f"v = {y}, D = {distances[x]+distance_y}")
					relax(y, distances[x]+distance_y)
					if(y in balls.keys()):
						for (z, distance_z) in balls[y]:
							#if(y != z):
								#print(f"z = {z}, node = {nodes[z]}")
								#print("relaxing z1,d(x)+wxy+d(y,z1)", f"v = {z}, D = {distances[x] + distance_y + calculate_distance(nodes[y], nodes[z], k)}")
								#print("relaxing z,d(x)+wxy+d(y,z)", f"v = {z}, D = {distances[x] + distance_y + distance_z}")
								#relax(z,distances[x] + distance_y + calculate_distance(nodes[y], nodes[z], k))
								relax(z,distances[x] + distance_y + distance_z)
	return distances, visited_count

def reconstruct_path(graph, distances, s, t,k,nodes):
	"""
	Reconstruct the path from source to target.

	:param graph: adjacency list of the graph
	:param distances: shortest distances from each node to s
	:param s: Source node
	:param t: Target node
	:return: List of nodes in the path
	"""
	path = [t]
	current = t
	last_path_length = len(path)
	while current is not s:
		
		#print('current = ', current)
		#print('path = ', path)
		for neighbor in graph[current]:
		#	print('checking neighbor', neighbor, distance_neighbor)
		#	print(distance_neighbor)
		#	print(distances[neighbor])
		#	print(distances[current])
			distance_neighbor = calculate_distance(nodes[current], nodes[neighbor],k)
			if (distances[neighbor] + distance_neighbor == distances[current]):
				if(neighbor not in path):
					path.append(neighbor)
					current = neighbor
					break

	path.reverse()

	# Check if there is a valid path from s to t
	if not path or path[0] != s:
		return None

	return path

def bundle_construction(graph, nodes, s, probability):
	R1 = [s]
	for node in nodes:
		if( random.random < probability):
			R1.append(node)
	R2 = []
	for v in nodes:
		if (v not in R1):
			#heap = FibonacciHeap()
			heap = HollowHeap()
			#heap.insert(0,v)
			heap.push(0, v)
			V = []
			#while heap.total_nodes > 0:
			while not heap.empty():
				#u = heap.extract_min()
				u = heap.find_min()
				heap.delete_min()
				V.append(u)
				if (u.key in R1):
					break
				elif len(V) > probability * np.log(probability):
					R2.append(v)
					break
				else:
					for x in graph[u]:
						if not heap.contains(x) and not x in V:
							#heap.insert(u.value + x[u.key], x)
							heap.push(u.item + x[u.key], x)
						else:
							#TODO
							break

def shortest_path(graph_path: str):
	'''
	Find the shortest path between two nodes in a graph using Dijkstra's algorithm.

	:param graph_path: Path to the file containing the graph data
	'''
	# Read and parse the graph
	n, m, k, s, t, nodes, edges, R = read_graph(graph_path)
	# Build the graph
	#R = {1.0: (-47.0, 55.0), 3.0: (-30.0, 48.0), 8.0: (77.0, -63.0), 12.0: (60.0, -16.0), 14.0: (10.0, 29.0), 19.0: (-56.0, 52.0), 20.0: (-24.0, -73.0), 21.0: (-72.0, -72.0), 24.0: (-28.0, -12.0), 25.0: (19.0, 76.0), 33.0: (43.0, -58.0), 37.0: (-62.0, 59.0), 38.0: (23.0, 98.0), 39.0: (-66.0, -100.0), 40.0: (68.0, -52.0), 42.0: (39.0, -23.0), 43.0: (-36.0, -80.0), 46.0: (83.0, -54.0), 47.0: (-50.0, 79.0), 50.0: (-5.0, -48.0), 52.0: (49.0, 55.0), 63.0: (5.0, 28.0), 64.0: (-66.0, 98.0), 67.0: (8.0, -35.0), 79.0: (7.0, -24.0), 80.0: (-30.0, -26.0), 82.0: (-8.0, 29.0), 85.0: (-41.0, -57.0), 86.0: (1.0, 67.0), 90.0: (29.0, -38.0), 91.0: (49.0, 26.0), 94.0: (41.0, 85.0), 98.0: (18.0, 34.0), 104.0: (-81.0, 98.0), 108.0: (-99.0, -61.0), 111.0: (29.0, -85.0), 112.0: (-88.0, 60.0), 118.0: (36.0, 78.0), 120.0: (57.0, 74.0), 121.0: (-51.0, -6.0), 124.0: (37.0, -15.0), 126.0: (-49.0, -80.0), 127.0: (-49.0, -4.0), 132.0: (65.0, 38.0), 133.0: (24.0, -55.0), 134.0: (-79.0, -94.0), 135.0: (-35.0, 3.0), 138.0: (14.0, -22.0), 140.0: (42.0, 31.0), 141.0: (-3.0, -63.0), 144.0: (-95.0, -77.0), 146.0: (-61.0, 9.0), 147.0: (-12.0, 0.0), 149.0: (16.0, -33.0), 150.0: (-79.0, 91.0), 154.0: (-46.0, -83.0), 159.0: (52.0, -66.0), 160.0: (-9.0, -35.0), 161.0: (48.0, -5.0), 163.0: (-2.0, -51.0), 165.0: (-31.0, 51.0), 168.0: (-88.0, 10.0), 169.0: (6.0, 79.0), 171.0: (35.0, 79.0), 172.0: (84.0, -95.0), 173.0: (-31.0, -2.0), 174.0: (-49.0, 63.0), 176.0: (49.0, -95.0), 177.0: (60.0, 46.0), 180.0: (-35.0, 12.0), 184.0: (-46.0, 96.0), 199.0: (-42.0, 54.0), 203.0: (54.0, 2.0), 208.0: (-44.0, 78.0), 209.0: (-35.0, 57.0), 213.0: (78.0, -72.0), 214.0: (13.0, -90.0), 215.0: (-23.0, 96.0), 216.0: (92.0, 80.0), 217.0: (98.0, 56.0), 219.0: (65.0, 62.0), 222.0: (-76.0, 8.0), 223.0: (65.0, 28.0), 229.0: (36.0, 39.0), 233.0: (7.0, -32.0), 234.0: (-64.0, -57.0), 236.0: (-93.0, 39.0), 237.0: (74.0, 94.0), 238.0: (-6.0, 99.0), 240.0: (-32.0, -72.0), 242.0: (61.0, 13.0), 245.0: (60.0, -20.0), 246.0: (-91.0, 14.0), 248.0: (-68.0, 72.0), 250.0: (-69.0, -92.0), 253.0: (-63.0, -76.0), 254.0: (72.0, 100.0), 257.0: (55.0, -85.0), 260.0: (-34.0, -76.0), 263.0: (-30.0, 46.0), 264.0: (92.0, 2.0), 268.0: (2.0, -62.0), 270.0: (-54.0, -48.0), 273.0: (81.0, 45.0), 275.0: (37.0, 17.0), 285.0: (12.0, 1.0), 286.0: (-30.0, 26.0), 289.0: (-3.0, -21.0), 291.0: (1.0, 28.0), 294.0: (-84.0, 35.0), 295.0: (28.0, 58.0), 296.0: (-100.0, -49.0), 304.0: (-30.0, 96.0), 306.0: (70.0, -67.0), 308.0: (-6.0, -53.0), 310.0: (-28.0, -7.0), 312.0: (-25.0, 95.0), 315.0: (-70.0, 91.0), 318.0: (74.0, 9.0), 321.0: (-94.0, -45.0), 323.0: (17.0, 59.0), 325.0: (-12.0, 73.0), 329.0: (8.0, -21.0), 334.0: (-91.0, 27.0), 336.0: (50.0, 20.0), 337.0: (67.0, 79.0), 339.0: (38.0, 86.0), 343.0: (59.0, 63.0), 344.0: (38.0, 9.0), 345.0: (17.0, 66.0), 346.0: (90.0, 23.0), 347.0: (-85.0, -89.0), 349.0: (-35.0, -58.0), 351.0: (96.0, -37.0), 353.0: (48.0, -45.0), 355.0: (38.0, 96.0), 367.0: (43.0, -76.0), 371.0: (-11.0, -93.0), 374.0: (13.0, -8.0), 378.0: (-83.0, -100.0), 381.0: (89.0, 51.0), 382.0: (92.0, 47.0), 384.0: (69.0, 33.0), 385.0: (9.0, 35.0), 386.0: (94.0, 12.0), 387.0: (34.0, -1.0), 390.0: (33.0, -10.0), 394.0: (99.0, 55.0), 395.0: (77.0, 65.0), 396.0: (-72.0, -64.0), 402.0: (19.0, -60.0), 407.0: (18.0, 4.0), 408.0: (-35.0, -4.0), 413.0: (29.0, -8.0), 414.0: (10.0, -72.0), 417.0: (73.0, 26.0), 418.0: (-2.0, 72.0), 422.0: (87.0, -66.0), 425.0: (37.0, 54.0), 427.0: (52.0, 90.0), 428.0: (-42.0, -55.0), 430.0: (59.0, 17.0), 431.0: (44.0, -30.0), 432.0: (-32.0, -1.0), 433.0: (64.0, -65.0), 434.0: (-1.0, 77.0), 437.0: (98.0, 66.0), 440.0: (-51.0, -32.0), 442.0: (-46.0, 73.0), 452.0: (-52.0, 99.0), 453.0: (29.0, 15.0), 455.0: (-60.0, 88.0), 456.0: (53.0, 77.0), 458.0: (95.0, -42.0), 464.0: (-35.0, 77.0), 466.0: (-63.0, 56.0), 468.0: (-59.0, -51.0), 470.0: (-79.0, 26.0), 471.0: (-93.0, 81.0), 476.0: (-5.0, 88.0), 481.0: (-19.0, 25.0), 483.0: (4.0, -92.0), 487.0: (66.0, 8.0), 494.0: (-96.0, 12.0), 496.0: (-58.0, -63.0), 497.0: (22.0, 99.0), 498.0: (89.0, 3.0), 500.0: (93.0, 60.0), 502.0: (-10000.0, 0.0)}

	print("build adjacency list")
	graph = build_graph(nodes, edges, k)
	print("bundle dijkstra")

	# Run Dijkstra's algorithm
	distances, visited_count = bundle_dijkstra(graph, s, R, k, nodes,edges)
	#print(distances)
	# Reconstruct the path
	#visualize_graph(nodes.values(), edges, R)
	print("reconstructing the path")
	path = reconstruct_path(graph, distances, s, t,k,nodes)

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


if __name__ == "__main__":
	import sys
	if len(sys.argv) > 1:
		import time
		start_time = time.time()
		shortest_path(sys.argv[1])
		print("--- %s seconds ---" % (time.time() - start_time))