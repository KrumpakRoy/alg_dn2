<<<<<<< HEAD
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
        distance = calculate_distance(nodes[u], nodes[v], k)
        
        # Add edges in both directions as the graph is undirected
        graph[u].append((v, distance))
        graph[v].append((u, distance))
    
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

def shortest_path(graph_path: str):
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

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        import time
        start_time = time.time()
        shortest_path(sys.argv[1])
=======
<<<<<<< HEAD
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
        distance = calculate_distance(nodes[u], nodes[v], k)
        
        # Add edges in both directions as the graph is undirected
        graph[u].append((v, distance))
        graph[v].append((u, distance))
    
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

def shortest_path(graph_path: str):
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

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        import time
        start_time = time.time()
        shortest_path(sys.argv[1])
=======
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
        distance = calculate_distance(nodes[u], nodes[v], k)
        
        # Add edges in both directions as the graph is undirected
        graph[u].append((v, distance))
        graph[v].append((u, distance))
    
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

def shortest_path(graph_path: str):
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

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        import time
        start_time = time.time()
        shortest_path(sys.argv[1])
>>>>>>> 417b1a8 (exploration code)
>>>>>>> d22471f (exploration code)
        print("--- %s seconds ---" % (time.time() - start_time))