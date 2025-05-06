<<<<<<< HEAD
import random
import networkx as nx
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

def make_fully_connected_edge_list(number_of_nodes, number_of_already_created_nodes=0):
	"""
	Generate a fully connected graph (as an edge list) with the given number of nodes.
	
	Args:
		number_of_nodes (int): The number of nodes in the graph.
	
	Returns:
		list: A list of tuples representing the edges in the graph.
	"""
	edges = list(combinations(range(number_of_already_created_nodes,number_of_already_created_nodes+number_of_nodes), 2))
	return edges

def make_fully_connected_graph_nodes(number_of_nodes,min_x, max_x, min_y, max_y, x_offset=0, y_offset=0):
	"""
	Create a fully connected graph with the given edges and node positions.
	
	Args:
		number_of_nodes (int): The number of nodes in the graph.
		min_x (int): Minimum x-coordinate for node positions.
		max_x (int): Maximum x-coordinate for node positions.
		min_y (int): Minimum y-coordinate for node positions.
		max_y (int): Maximum y-coordinate for node positions.
	
	Returns:
		list: A list of dictionaries representing the nodes in the graph.
	"""
	nodes = np.column_stack((
		np.random.randint(min_x+x_offset, max_x + 1+x_offset, size=number_of_nodes),
		np.random.randint(min_y+y_offset, max_y + 1+y_offset, size=number_of_nodes)
	)).tolist()
	return nodes

def visualize_graph(nodes, edges):
	plt.figure(figsize=(10, 8))
	x = [node[0] for node in nodes]
	y = [node[1] for node in nodes]

	plt.scatter(x, y, c='blue', s=100)

	# Plot edges
	#print(edges)
	for edge in edges:
		u = edge[0]
		v = edge[1]
		plt.plot([x[u], x[v]], [y[u], y[v]], 'k-', alpha=0.6)

	# Add node labels
	for i, (node_x, node_y) in enumerate(nodes):
		plt.text(node_x, node_y, str(i), fontsize=12, ha='center', va='center')

	plt.grid(True)
	plt.axis('equal')
	plt.show()
def create_graph(number_of_fully_connected_subgraphs, number_of_all_nodes):
	"""
	Create a graph of fully connected subgraphs, connected by a single edge.
	"""
	edges = []
	nodes = []
	subgraph_size = number_of_all_nodes // number_of_fully_connected_subgraphs
	for i in range(number_of_fully_connected_subgraphs):
		subgraph_edges = make_fully_connected_edge_list(subgraph_size, i * subgraph_size)
		subgraph_nodes = make_fully_connected_graph_nodes(subgraph_size, -100, 100, -100, 100, i * 500)
		edges.extend(subgraph_edges)
		nodes.extend(subgraph_nodes)

	# Connect the last node of the previous subgraph to the first node of the next subgraph
	for i in range(0,number_of_fully_connected_subgraphs - 1):
		edges.append(((i + 1) * subgraph_size - 1, (i + 1) * subgraph_size))
	return nodes, edges

total_number_of_nodes = 3
number_of_subgraphs = 1
size_of_subgraph = total_number_of_nodes // number_of_subgraphs

nodes, edges = create_graph(number_of_subgraphs, total_number_of_nodes)


s = (-10000, 0)
m = (10000, 0)
t = (10000, 10000)
nodes.append(m)
nodes.append(s)
nodes.append(t)

random_in_node =  random.randint((number_of_subgraphs-1)*size_of_subgraph,total_number_of_nodes)

edge_m_graph = (total_number_of_nodes,random_in_node)
edge_s_graph = (total_number_of_nodes+1, 0)
edge_m_t = (total_number_of_nodes, total_number_of_nodes+2)
edges.append(edge_m_graph)
edges.append(edge_s_graph)
edges.append(edge_m_t)


#visualize_graph(nodes, edges)
with open(f'test_graph_{number_of_subgraphs}_subgraphs_{total_number_of_nodes}.txt', 'w') as f:
	f.write(f"{total_number_of_nodes+3} {len(edges)} 2 {total_number_of_nodes+2} {total_number_of_nodes+3}\n")
	for i, node in enumerate(nodes):
		f.write(f"{i+1} {node[0]} {node[1]}\n")
	for edge in edges:
=======
<<<<<<< HEAD
import random
import networkx as nx
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

def make_fully_connected_edge_list(number_of_nodes, number_of_already_created_nodes=0):
	"""
	Generate a fully connected graph (as an edge list) with the given number of nodes.
	
	Args:
		number_of_nodes (int): The number of nodes in the graph.
	
	Returns:
		list: A list of tuples representing the edges in the graph.
	"""
	edges = list(combinations(range(number_of_already_created_nodes,number_of_already_created_nodes+number_of_nodes), 2))
	return edges

def make_fully_connected_graph_nodes(number_of_nodes,min_x, max_x, min_y, max_y, x_offset=0, y_offset=0):
	"""
	Create a fully connected graph with the given edges and node positions.
	
	Args:
		number_of_nodes (int): The number of nodes in the graph.
		min_x (int): Minimum x-coordinate for node positions.
		max_x (int): Maximum x-coordinate for node positions.
		min_y (int): Minimum y-coordinate for node positions.
		max_y (int): Maximum y-coordinate for node positions.
	
	Returns:
		list: A list of dictionaries representing the nodes in the graph.
	"""
	nodes = np.column_stack((
		np.random.randint(min_x+x_offset, max_x + 1+x_offset, size=number_of_nodes),
		np.random.randint(min_y+y_offset, max_y + 1+y_offset, size=number_of_nodes)
	)).tolist()
	return nodes

def visualize_graph(nodes, edges):
	plt.figure(figsize=(10, 8))
	x = [node[0] for node in nodes]
	y = [node[1] for node in nodes]

	plt.scatter(x, y, c='blue', s=100)

	# Plot edges
	#print(edges)
	for edge in edges:
		u = edge[0]
		v = edge[1]
		plt.plot([x[u], x[v]], [y[u], y[v]], 'k-', alpha=0.6)

	# Add node labels
	for i, (node_x, node_y) in enumerate(nodes):
		plt.text(node_x, node_y, str(i), fontsize=12, ha='center', va='center')

	plt.grid(True)
	plt.axis('equal')
	plt.show()
def create_graph(number_of_fully_connected_subgraphs, number_of_all_nodes):
	"""
	Create a graph of fully connected subgraphs, connected by a single edge.
	"""
	edges = []
	nodes = []
	subgraph_size = number_of_all_nodes // number_of_fully_connected_subgraphs
	for i in range(number_of_fully_connected_subgraphs):
		subgraph_edges = make_fully_connected_edge_list(subgraph_size, i * subgraph_size)
		subgraph_nodes = make_fully_connected_graph_nodes(subgraph_size, -100, 100, -100, 100, i * 500)
		edges.extend(subgraph_edges)
		nodes.extend(subgraph_nodes)

	# Connect the last node of the previous subgraph to the first node of the next subgraph
	for i in range(0,number_of_fully_connected_subgraphs - 1):
		edges.append(((i + 1) * subgraph_size - 1, (i + 1) * subgraph_size))
	return nodes, edges

total_number_of_nodes = 3
number_of_subgraphs = 1
size_of_subgraph = total_number_of_nodes // number_of_subgraphs

nodes, edges = create_graph(number_of_subgraphs, total_number_of_nodes)


s = (-10000, 0)
m = (10000, 0)
t = (10000, 10000)
nodes.append(m)
nodes.append(s)
nodes.append(t)

random_in_node =  random.randint((number_of_subgraphs-1)*size_of_subgraph,total_number_of_nodes)

edge_m_graph = (total_number_of_nodes,random_in_node)
edge_s_graph = (total_number_of_nodes+1, 0)
edge_m_t = (total_number_of_nodes, total_number_of_nodes+2)
edges.append(edge_m_graph)
edges.append(edge_s_graph)
edges.append(edge_m_t)


#visualize_graph(nodes, edges)
with open(f'test_graph_{number_of_subgraphs}_subgraphs_{total_number_of_nodes}.txt', 'w') as f:
	f.write(f"{total_number_of_nodes+3} {len(edges)} 2 {total_number_of_nodes+2} {total_number_of_nodes+3}\n")
	for i, node in enumerate(nodes):
		f.write(f"{i+1} {node[0]} {node[1]}\n")
	for edge in edges:
=======
import random
import networkx as nx
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

def make_fully_connected_edge_list(number_of_nodes, number_of_already_created_nodes=0):
	"""
	Generate a fully connected graph (as an edge list) with the given number of nodes.
	
	Args:
		number_of_nodes (int): The number of nodes in the graph.
	
	Returns:
		list: A list of tuples representing the edges in the graph.
	"""
	edges = list(combinations(range(number_of_already_created_nodes,number_of_already_created_nodes+number_of_nodes), 2))
	return edges

def make_fully_connected_graph_nodes(number_of_nodes,min_x, max_x, min_y, max_y, x_offset=0, y_offset=0):
	"""
	Create a fully connected graph with the given edges and node positions.
	
	Args:
		number_of_nodes (int): The number of nodes in the graph.
		min_x (int): Minimum x-coordinate for node positions.
		max_x (int): Maximum x-coordinate for node positions.
		min_y (int): Minimum y-coordinate for node positions.
		max_y (int): Maximum y-coordinate for node positions.
	
	Returns:
		list: A list of dictionaries representing the nodes in the graph.
	"""
	nodes = np.column_stack((
		np.random.randint(min_x+x_offset, max_x + 1+x_offset, size=number_of_nodes),
		np.random.randint(min_y+y_offset, max_y + 1+y_offset, size=number_of_nodes)
	)).tolist()
	return nodes

def visualize_graph(nodes, edges):
	plt.figure(figsize=(10, 8))
	x = [node[0] for node in nodes]
	y = [node[1] for node in nodes]

	plt.scatter(x, y, c='blue', s=100)

	# Plot edges
	#print(edges)
	for edge in edges:
		u = edge[0]
		v = edge[1]
		plt.plot([x[u], x[v]], [y[u], y[v]], 'k-', alpha=0.6)

	# Add node labels
	for i, (node_x, node_y) in enumerate(nodes):
		plt.text(node_x, node_y, str(i), fontsize=12, ha='center', va='center')

	plt.grid(True)
	plt.axis('equal')
	plt.show()
def create_graph(number_of_fully_connected_subgraphs, number_of_all_nodes):
	"""
	Create a graph of fully connected subgraphs, connected by a single edge.
	"""
	edges = []
	nodes = []
	subgraph_size = number_of_all_nodes // number_of_fully_connected_subgraphs
	for i in range(number_of_fully_connected_subgraphs):
		subgraph_edges = make_fully_connected_edge_list(subgraph_size, i * subgraph_size)
		subgraph_nodes = make_fully_connected_graph_nodes(subgraph_size, -100, 100, -100, 100, i * 500)
		edges.extend(subgraph_edges)
		nodes.extend(subgraph_nodes)

	# Connect the last node of the previous subgraph to the first node of the next subgraph
	for i in range(0,number_of_fully_connected_subgraphs - 1):
		edges.append(((i + 1) * subgraph_size - 1, (i + 1) * subgraph_size))
	return nodes, edges

total_number_of_nodes = 3
number_of_subgraphs = 1
size_of_subgraph = total_number_of_nodes // number_of_subgraphs

nodes, edges = create_graph(number_of_subgraphs, total_number_of_nodes)


s = (-10000, 0)
m = (10000, 0)
t = (10000, 10000)
nodes.append(m)
nodes.append(s)
nodes.append(t)

random_in_node =  random.randint((number_of_subgraphs-1)*size_of_subgraph,total_number_of_nodes)

edge_m_graph = (total_number_of_nodes,random_in_node)
edge_s_graph = (total_number_of_nodes+1, 0)
edge_m_t = (total_number_of_nodes, total_number_of_nodes+2)
edges.append(edge_m_graph)
edges.append(edge_s_graph)
edges.append(edge_m_t)


#visualize_graph(nodes, edges)
with open(f'test_graph_{number_of_subgraphs}_subgraphs_{total_number_of_nodes}.txt', 'w') as f:
	f.write(f"{total_number_of_nodes+3} {len(edges)} 2 {total_number_of_nodes+2} {total_number_of_nodes+3}\n")
	for i, node in enumerate(nodes):
		f.write(f"{i+1} {node[0]} {node[1]}\n")
	for edge in edges:
>>>>>>> 417b1a8 (exploration code)
>>>>>>> d22471f (exploration code)
		f.write(f"{edge[0]+1} {edge[1]+1}\n")