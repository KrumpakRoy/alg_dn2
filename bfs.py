<<<<<<< HEAD
import sys
import math
from collections import deque
import time

def read_graph_from_file(file_path):
    """Read graph data from file and return it as structured data."""
    with open(file_path, 'r') as f:
        # Read first line with n, m, k, s, t
        n, m, k, s, t = map(int, f.readline().strip().split())
        
        # Read node coordinates
        coordinates = [None] * (n+1)  # 1-indexed array for coordinates
        for _ in range(n):
            line = f.readline().strip().split()
            node_id = int(line[0])
            x, y = float(line[1]), float(line[2])
            coordinates[node_id] = (x, y)
        
        # Read adjacency data
        adjacency_list = [[] for _ in range(n+1)]  # 1-indexed
        for _ in range(m):
            u, v = map(int, f.readline().strip().split())
            # Add edges in both directions (undirected graph)
            adjacency_list[u].append(v)
            adjacency_list[v].append(u)
            
    return n, m, k, s, t, coordinates, adjacency_list

def calculate_distance(coord1, coord2):
    """Calculate Euclidean distance between two coordinates."""
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def bfs_shortest_path(adjacency_list, coordinates, start, end, n):
    """
    Find shortest path using BFS.
    Returns number of visited nodes, total path length, and the path itself.
    """
    # Initialize
    visited = [False] * (n+1)  # 1-indexed
    queue = deque([(start, [start], 0)])  # (current_node, path_so_far, path_length)
    visited[start] = True
    visited_count = 1
    
    while queue:
        current, path, length = queue.popleft()
        
        if current == end:
            return visited_count, length, path
        
        for neighbor in adjacency_list[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                visited_count += 1
                
                # Calculate distance between current node and neighbor
                edge_length = calculate_distance(coordinates[current], coordinates[neighbor])
                
                # Add neighbor to queue with updated path and length
                new_path = path + [neighbor]
                new_length = length + edge_length
                
                queue.append((neighbor, new_path, new_length))
    
    # If no path is found
    return visited_count, float('inf'), []

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_graph_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Read graph data
    n, m, k, s, t, coordinates, adjacency_list = read_graph_from_file(file_path)
    
    # Find shortest path using BFS
    visited_count, path_length, path = bfs_shortest_path(adjacency_list, coordinates, s, t, n)
    
    # Output results
    print(visited_count)
    print(f"{path_length:.6f}")
    print(" ".join(map(str, path)))

if __name__ == "__main__":
    time_start = time.time()
    main()
=======
import sys
import math
from collections import deque
import time

def read_graph_from_file(file_path):
    """Read graph data from file and return it as structured data."""
    with open(file_path, 'r') as f:
        # Read first line with n, m, k, s, t
        n, m, k, s, t = map(int, f.readline().strip().split())
        
        # Read node coordinates
        coordinates = [None] * (n+1)  # 1-indexed array for coordinates
        for _ in range(n):
            line = f.readline().strip().split()
            node_id = int(line[0])
            x, y = float(line[1]), float(line[2])
            coordinates[node_id] = (x, y)
        
        # Read adjacency data
        adjacency_list = [[] for _ in range(n+1)]  # 1-indexed
        for _ in range(m):
            u, v = map(int, f.readline().strip().split())
            # Add edges in both directions (undirected graph)
            adjacency_list[u].append(v)
            adjacency_list[v].append(u)
            
    return n, m, k, s, t, coordinates, adjacency_list

def calculate_distance(coord1, coord2):
    """Calculate Euclidean distance between two coordinates."""
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def bfs_shortest_path(adjacency_list, coordinates, start, end, n):
    """
    Find shortest path using BFS.
    Returns number of visited nodes, total path length, and the path itself.
    """
    # Initialize
    visited = [False] * (n+1)  # 1-indexed
    queue = deque([(start, [start], 0)])  # (current_node, path_so_far, path_length)
    visited[start] = True
    visited_count = 1
    
    while queue:
        current, path, length = queue.popleft()
        
        if current == end:
            return visited_count, length, path
        
        for neighbor in adjacency_list[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                visited_count += 1
                
                # Calculate distance between current node and neighbor
                edge_length = calculate_distance(coordinates[current], coordinates[neighbor])
                
                # Add neighbor to queue with updated path and length
                new_path = path + [neighbor]
                new_length = length + edge_length
                
                queue.append((neighbor, new_path, new_length))
    
    # If no path is found
    return visited_count, float('inf'), []

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_graph_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Read graph data
    n, m, k, s, t, coordinates, adjacency_list = read_graph_from_file(file_path)
    
    # Find shortest path using BFS
    visited_count, path_length, path = bfs_shortest_path(adjacency_list, coordinates, s, t, n)
    
    # Output results
    print(visited_count)
    print(f"{path_length:.6f}")
    print(" ".join(map(str, path)))

if __name__ == "__main__":
    time_start = time.time()
    main()
>>>>>>> 417b1a8 (exploration code)
    print(f"Elapsed time: {time.time() - time_start:.3f} seconds")