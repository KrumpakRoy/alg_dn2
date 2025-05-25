from a_star import a_star_search
from bfs import bfs_shortest_path
from dijkstra import dijkstra, build_graph, reconstruct_path
from fibonacci_heap import shortest_path_dijkstra_fibonacci, build_adjacency_list
from bi_dijkstra import bi_dijkstra as shortest_path_bi_dijkstra
import numpy as np
from generate_graph import get_graph
import time
import pandas as pd

def time_function(function, *arguments):
	start_time = time.time()
	number_of_visited_nodes, path_length, path = function(*arguments)
	#print(number_of_visited_nodes)
	#print(path_length)
	#print(path)
	execution_time = time.time()-start_time
	#print(f"elapsed time: {execution_time:.2} seconds")
	return number_of_visited_nodes, path_length, path, execution_time

max_number_of_subgraphs = 10
range_number_of_nodes = np.ceil(np.logspace(1,4,20))

the_best = pd.DataFrame(data=[],columns =range_number_of_nodes)
#functions = [a_star_search, dijkstra, bfs_shortest_path, shortest_path_dijkstra_fibonacci]
functions = [shortest_path_bi_dijkstra, dijkstra, a_star_search]
for number_of_subgraphs in range(1, max_number_of_subgraphs):
	row_best = pd.DataFrame(data=np.atleast_2d(np.zeros(len(range_number_of_nodes))),columns =range_number_of_nodes)
	for number_of_nodes in range_number_of_nodes:
		print("numer of subgraphs: ", number_of_subgraphs, "number of nodes: ", number_of_nodes)
		k = 2
		number_of_nodes = int(number_of_nodes)
		number_of_nodes = number_of_nodes + number_of_nodes % number_of_subgraphs
		nodes, edges, s, t = get_graph(number_of_nodes, number_of_subgraphs)
		
		#print(len(nodes))
		adjacency_list = build_adjacency_list(len(nodes), edges)
		nodes_dict = {i+1: node for i, node in enumerate(nodes)}
		
		graph = build_graph(nodes_dict, edges, k)

		num_visited_a, path_length_a, path_a, exec_time_a = time_function(functions[2], adjacency_list,nodes_dict, s, t, k)
		dijkstra_results = time_function(functions[1], graph, s, t)
		dijkstra_path = reconstruct_path(dijkstra_results[2],s,t)
		num_visited, path_length, path, exec_time = time_function(functions[0], graph, s, t)
		try: 
			assert path_length == dijkstra_results[1][t] and path == dijkstra_path
			assert path_length_a == dijkstra_results[1][t] and path_a == dijkstra_path
		except:
			print("the paths do not match")
			print(num_visited, path_length, path)
			print(num_visited_a, path_length_a, path_a)
			print(dijkstra_results[0], dijkstra_results[1][t], dijkstra_path)
		
		if (num_visited == 0 or path == []):
			print('fail, the graph wasn\'t connected')
			if(dijkstra_results[2] != []):
				print("dijkstra finished, but bi dijkstra did not")
		#	nodes, edges, s, t = get_graph(number_of_nodes, number_of_subgraphs)
		elif(dijkstra_results[3] < exec_time):
			print("dijkstra")
		else:
			print("bidirectional dijkstra")


		
		#try: 
		#	assert path_length == dijkstra_results[1][t] and path == dijkstra_path
		#except:
		#	print("dijkstra wrong")
		#	print(num_visited, path_length, path)
		#	print(dijkstra_results[0], dijkstra_results[1][t], dijkstra_path)
#
		##print("coordinates in tests:",[].extend(nodes))
		#bfs_results = time_function(functions[2], adjacency_list, nodes_dict, s, t, len(nodes),k)
		#try: 
		#	assert path_length == bfs_results[1] and path == bfs_results[2]
		#except:
		#	print("BFS wrong")
		#	print(num_visited, path_length, path)
		#	print(bfs_results)
#
#
		#fib_results = time_function(functions[3], adjacency_list, s,t, nodes_dict,k)
		##print("sakdfjasjd",fib_results)
		#try: 
		#	assert path_length == fib_results[1] and path == fib_results[2]
		#except:
		#	print("fibonacci heap wrong")
		#	print(num_visited, path_length, path)
		#	print(fib_results)
#
		#row_best.loc[0,number_of_nodes]=np.argmin([exec_time, dijkstra_results[3], bfs_results[3],fib_results[3]])
		row_best.loc[0,number_of_nodes]=np.argmin([exec_time, dijkstra_results[3], exec_time_a])
	
	the_best = pd.concat([the_best, row_best], axis = 0)

#the_best.to_csv("the_best_option.csv")
print(dict(zip(*np.unique(the_best, return_counts=True))))
