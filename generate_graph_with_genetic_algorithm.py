import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from collections import deque
import copy
import time
from shortest_path import shortest_path
import sys
import heapq
from pathlib import Path
from typing import Dict, List, Tuple, Optional
class GraphGenetic:
	def __init__(self, V=10, k=2, n=50, p=0.1,c=0.01, N=20, P=0.3, O=0.1):
		"""
		Initialize the genetic algorithm for evolving graphs.
		
		Parameters:
		V: Number of vertices in each graph
		k: Parameter determining how distances are calculated between nodes
		n: Number of iterations/generations
		p: Probability of mutation
		c: Cooldown for mutation probability
		N: Population size
		P: Percentage of top instances that will reproduce
		O: Probability of other instances to be reproduced
		"""
		self.V = V
		self.k = k
		self.n = n
		self.p = p
		self.mutation_cooldown = c
		self.N = N
		self.P = P
		self.O = O
		
		# Select start and end points
		self.start_point = (random.uniform(-1000, 1000), random.uniform(-1000, 1000))
		self.end_point = (random.uniform(-1000, 1000), random.uniform(-1000, 1000))
		
		# Population of graphs
		self.population = []
		
		# Best graph found so far
		self.best_graph = None
		self.best_fitness = 0
		
		# History for plotting
		self.avg_fitness_history = []
		self.best_fitness_history = []

	def initialize_population(self):
		"""Generate initial population of random connected graphs"""
		self.population = []
		
		for _ in range(self.N):
			# Create a new graph
			graph = {'nodes': [], 'edges': []}
			
			# Add the start and end points
			graph['nodes'].append(self.start_point)
			graph['nodes'].append(self.end_point)
			
			# Generate remaining nodes randomly
			for _ in range(self.V - 2):
				x = random.uniform(-1000, 1000)
				y = random.uniform(-1000, 1000)
				graph['nodes'].append((x, y))
			
			# Create a connected graph
			self._make_connected_graph(graph)
			
			self.population.append(graph)

	def _make_connected_graph(self, graph):
		"""Ensure the graph is connected by creating a spanning tree and then adding random edges"""
		nodes = graph['nodes']
		n = len(nodes)
		
		# Start with an empty edge set
		graph['edges'] = []
		
		# Create a minimum spanning tree to ensure connectivity
		visited = [1]  # Start with node 1
		unvisited = list(range(2, n+1))
		
		while unvisited:
			source = random.choice(visited)
			target = random.choice(unvisited)
			unvisited.remove(target)
			visited.append(target)
			graph['edges'].append((min(source, target), max(source, target)))
		
		# Add some random edges
		for _ in range(random.randint(0, n)):
			u = random.randint(1, n)
			v = random.randint(1, n)
			if u != v:
				graph['edges'].append((min(u, v), max(u, v)))

	def calculate_distance(self, point1, point2):
		"""Calculate distance between two points based on parameter k"""
		x1, y1 = point1
		x2, y2 = point2
		
		if self.k == 1:
			# Manhattan distance
			return abs(x1 - x2) + abs(y1 - y2)
		elif self.k == 2:
			# Euclidean distance
			return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
		else:
			# Minkowski distance
			return ((abs(x1 - x2)**self.k) + (abs(y1 - y2)**self.k))**(1/self.k)

	def _edge_len(self, dx: float, dy: float, k: int) -> float:
		"""Return D(u,v,k) for given coordinate differences and integer k."""

		# special case (every edge has weight 1)
		if k == 0:
			return 1.0
		# Chebyshev distance    
		if k == -1:
			return max(abs(dx), abs(dy))
		#  k  >=  :  Minkowski distance
		return (abs(dx) ** k + abs(dy) ** k) ** (1.0 / k)
	
	def shortest_path(self, graph) -> None:
		"""
		Read the graph stored in *graph_path*, compute the shortest s-t walk
		and print:
			1) number of vertex-expansions
			2) length of the shortest walk |p|
			3) vertex id sequence of the walk (space-separated)
		"""
		coords = {i+1: node for i, node in enumerate(graph['nodes'])}
		edges = graph['edges']
		#print(edges)
		s = 1
		t = 2
		adj: Dict[int, List[Tuple[int, float]]] = {v: [] for v in coords}
		for edge in edges:
			u = edge[0]
			v = edge[1]
			#print(u, v)
			dx, dy = coords[u][0] - coords[v][0], coords[u][1] - coords[v][1]
			w = self._edge_len(dx, dy, self.k)
			#print(adj)
			adj[u].append((v, w))
			adj[v].append((u, w))
		# ---------- 2. Dijkstra ----------------------------------------------------
		INF = float("inf")
		dist: Dict[int, float] = {v: INF for v in coords}
		prev: Dict[int, Optional[int]] = {v: None for v in coords}
		dist[s] = 0.0

		heap: List[Tuple[float, int]] = [(0.0, s)]
		visited_count = 0

		while heap:
			d, u = heapq.heappop(heap)
			visited_count += 1                      # count every expansion
			if d != dist[u]:
				continue                            # stale entry
			if u == t:
				break
			for v, w in adj[u]:
				nd = d + w
				if nd < dist[v] - 1e-12:           # avoid fp flapping
					dist[v] = nd
					prev[v] = u
					heapq.heappush(heap, (nd, v))

		# ---------- 3. rebuild path ------------------------------------------------
		if dist[t] == INF:
			raise ValueError("Graph is not connected - no s-t path found.")

		path: List[int] = []
		cur = t
		while cur is not None:
			path.append(cur)
			cur = prev[cur]
		path.reverse()
		return visited_count

	def evaluate_fitness(self, graph):
		"""Evaluate fitness of a graph based on shortest path length"""
		path_length = self.shortest_path(graph)
		return path_length

	def select_parents(self):
		"""Select parents for reproduction based on their fitness"""
		# Calculate fitness for each graph
		fitness_values = [self.evaluate_fitness(graph) for graph in self.population]
		
		# Sort population by fitness (higher is better)
		sorted_indices = np.argsort(fitness_values)[::-1]
		sorted_population = [self.population[i] for i in sorted_indices]
		sorted_fitness = [fitness_values[i] for i in sorted_indices]
		
		# Update best graph if found
		if sorted_fitness[0] > self.best_fitness:
			self.best_fitness = sorted_fitness[0]
			self.best_graph = copy.deepcopy(sorted_population[0])
		
		# Record history
		self.avg_fitness_history.append(np.mean(fitness_values))
		self.best_fitness_history.append(sorted_fitness[0])
		
		# Select top P% for reproduction
		top_count = max(2, int(self.P * self.N))
		parents = sorted_population[:top_count]
		
		# Randomly select other individuals with probability O
		for i in range(top_count, len(sorted_population)):
			if random.random() < self.O:
				parents.append(sorted_population[i])
		
		return parents

	def crossover(self, parent1, parent2):
		"""Perform crossover between two parent graphs"""
		# Create two new children
		child1 = {'nodes': [], 'edges': set()}
		child2 = {'nodes': [], 'edges': set()}
		
		# Always keep start and end points
		child1['nodes'] = [parent1['nodes'][0], parent1['nodes'][1]]
		child2['nodes'] = [parent2['nodes'][0], parent2['nodes'][1]]
		
		# Randomly select crossover point
		crossover_point = random.randint(2, self.V - 1)
		
		# Add nodes from parents to children
		child1['nodes'].extend(parent1['nodes'][2:crossover_point])
		child1['nodes'].extend(parent2['nodes'][crossover_point:])
		
		child2['nodes'].extend(parent2['nodes'][2:crossover_point])
		child2['nodes'].extend(parent1['nodes'][crossover_point:])
		
		# Create new edges for each child
		self._make_connected_graph(child1)
		self._make_connected_graph(child2)
		
		return child1, child2

	def mutate(self, graph):
		"""Apply mutation to a graph with probability p"""
		if random.random() > self.p:
			return graph
		
		# Create a deep copy to avoid modifying the original
		mutated = copy.deepcopy(graph)
		
		# Choose mutation type
		mutation_type = random.choice(['node', 'edge'])
		
		if mutation_type == 'node':
			# Move a random node (except start and end)
			if len(mutated['nodes']) > 2:
				node_idx = random.randint(2, len(mutated['nodes']) - 1)
				x = random.uniform(0, 10)
				y = random.uniform(0, 10)
				mutated['nodes'][node_idx] = (x, y)
		else:
			# Add or remove an edge
			edges_list = list(mutated['edges'])
			
			if random.random() < 0.5 and edges_list:
				# Remove a random edge if there are edges to remove
				edge_to_remove = random.choice(edges_list)
				
				# Check if removing this edge would disconnect the graph
				temp_edges = mutated['edges'].copy()
				temp_edges.remove(edge_to_remove)
				
				# Create a graph representation to check connectivity
				G = nx.Graph()
				G.add_nodes_from(range(len(mutated['nodes'])))
				G.add_edges_from(temp_edges)
				
				# Only remove if the graph remains connected
				if nx.is_connected(G):
					mutated['edges'] = temp_edges
			else:
				# Add a random edge
				n = len(mutated['nodes'])
				u = random.randint(1, n)
				v = random.randint(1, n)
				
				if u != v and (min(u, v), max(u, v)) not in mutated['edges']:
					mutated['edges'].append((min(u, v), max(u, v)))
		
		return mutated

	def evolve(self):
		"""Run the genetic algorithm for n iterations"""
		# Initialize population
		self.initialize_population()
		
		for iteration in range(self.n):
			# Select parents
			parents = self.select_parents()
			
			# Create new population through crossover and mutation
			new_population = []
			
			# Elitism: keep the best solution
			new_population.append(copy.deepcopy(self.best_graph))
			
			while len(new_population) < self.N:
				parent1 = random.choice(parents)
				parent2 = random.choice(parents)
				
				child1, child2 = self.crossover(parent1, parent2)
				
				# Apply mutation
				child1 = self.mutate(child1)
				child2 = self.mutate(child2)
				
				new_population.append(child1)
				new_population.append(child2)
			
			# Truncate if we have too many
			self.population = new_population[:self.N]
			
			# Print progress
			if (iteration + 1) % 10 == 0 or iteration == 0:
				print(f"Iteration {iteration + 1}/{self.n}, Best Fitness: {self.best_fitness}, number of edges: {len(self.best_graph['edges'])}")
				self.p = max(0.01, self.p - self.mutation_cooldown)  # Decrease mutation probability over time

	def visualize_best(self):
		"""Visualize the best graph found"""
		if self.best_graph is None:
			print("No best graph found yet.")
			return
			
		plt.figure(figsize=(10, 8))
		
		# Plot nodes
		nodes = self.best_graph['nodes']
		x = [node[0] for node in nodes]
		y = [node[1] for node in nodes]
		
		plt.scatter(x, y, c='blue', s=100)
		
		# Highlight start and end nodes
		plt.scatter([nodes[0][0]], [nodes[0][1]], c='green', s=200, label='Start')
		plt.scatter([nodes[1][0]], [nodes[1][1]], c='red', s=200, label='End')
		
		# Plot edges
		for edge in self.best_graph['edges']:
			u = edge[0] - 1
			v = edge[1] - 1
			plt.plot([x[u], x[v]], [y[u], y[v]], 'k-', alpha=0.6)
		
		# Add node labels
		for i, (node_x, node_y) in enumerate(nodes):
			plt.text(node_x, node_y, str(i), fontsize=12, ha='center', va='center')
		
		plt.title(f"Best Graph (Shortest Path: {self.best_fitness} nodes)")
		plt.legend()
		plt.grid(True)
		plt.axis('equal')
		plt.show()

	def visualize_progress(self):
		"""Visualize the progress of the genetic algorithm"""
		plt.figure(figsize=(10, 6))
		
		iterations = range(1, len(self.avg_fitness_history) + 1)
		plt.plot(iterations, self.avg_fitness_history, 'b-', label='Average Fitness')
		plt.plot(iterations, self.best_fitness_history, 'r-', label='Best Fitness')
		
		plt.xlabel('Iteration')
		plt.ylabel('Fitness (Shortest Path Length)')
		plt.title('Genetic Algorithm Progress')
		plt.legend()
		plt.grid(True)
		plt.show()

def main():
	# Parse command line arguments
	import argparse
	parser = argparse.ArgumentParser(description='Genetic Algorithm for Graph Evolution')
	parser.add_argument('--vertices', '-V', type=int, default=10, help='Number of vertices')
	parser.add_argument('--k', type=int, default=2, help='Parameter for distance calculation')
	parser.add_argument('--iterations', '-n', type=int, default=50, help='Number of iterations')
	parser.add_argument('--mutation_prob', '-p', type=float, default=0.5, help='Mutation probability')
	parser.add_argument('--mutation_cooldown', '-c', type=float, default=0.01, help='Cooldown for mutation')
	parser.add_argument('--population_size', '-N', type=int, default=20, help='Population size')
	parser.add_argument('--top_percent', '-P', type=float, default=0.3, help='Percentage of top instances to reproduce')
	parser.add_argument('--other_prob', '-O', type=float, default=0.1, help='Probability of other instances to reproduce')

	args = parser.parse_args()

	# Run the genetic algorithm
	start_time = time.time()

	ga = GraphGenetic(
		V=args.vertices,
		k=args.k,
		n=args.iterations,
		p=args.mutation_prob,
		N=args.population_size,
		P=args.top_percent,
		O=args.other_prob
	)

	print(f"Start point: {ga.start_point}")
	print(f"End point: {ga.end_point}")
	print(f"Running genetic algorithm with parameters:")
	print(f"  Vertices: {args.vertices}")
	print(f"  Distance parameter k: {args.k}")
	print(f"  Iterations: {args.iterations}")
	print(f"  Mutation probability: {args.mutation_prob}")
	print(f"  Population size: {args.population_size}")
	print(f"  Top percent for reproduction: {args.top_percent}")
	print(f"  Other instances probability: {args.other_prob}")

	ga.evolve()

	end_time = time.time()
	print(f"Time taken: {end_time - start_time:.2f} seconds")

	print(f"Best path length: {ga.best_fitness}")

	# Visualize results
	ga.visualize_best()
	ga.visualize_progress()

if __name__ == "__main__":
	main()
