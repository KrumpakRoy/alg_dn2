import subprocess
import time
import pandas as pd
import os
import functools

#graph_paths =[ 
#			  "inst1_comb_k0.txt",
#			  "test_graph_1_subgraphs_100.txt",
#			  "inst2_block_cheb.txt",
#			  "test_graph_1_subgraphs_1024.txt",
#			  "inst3_spiral.txt",
#			  "inst4_tcomb.txt",
#			  "test_graph_1_subgraphs_500.txt",
#			  "inst5_flower.txt",
#			  "test_graph_2_subgraphs_5000.txt",
#			  "test_graph_3_subgraphs_9600.txt",
#			  ]

folder = 'grafi/'
subfolders = os.listdir(folder)
graph_paths = [os.listdir(folder+subfolder) for subfolder in subfolders]
graph_paths = [[subfolders[i]+"/"+graph for graph in graphs] for i,graphs in enumerate(graph_paths)]
graph_paths = functools.reduce(lambda a, b: a + b, graph_paths)
graph_paths = [folder+path for path in graph_paths]
print(graph_paths)
print(len(graph_paths))


py_files = ["first_implementations/1.py"]+[f"first_implementations/{x}.py" for x in range(3,25)]
times = pd.DataFrame(data =[], columns=py_files+["graph"])

absolute_start_time = time.time()
for i,graph in enumerate(graph_paths):
	times.loc[i,"graph"] = graph
	print("graph: ", graph)
	for script in py_files:
		print(f"\nRunning {script}...")
		start = time.time()
		timeout_seconds = 300
		try:
			result = subprocess.run(["python", script, graph], capture_output=True, text=True, timeout=timeout_seconds)
		except subprocess.TimeoutExpired:
			print(f"{script} timed out after {timeout_seconds} seconds.")
		end = time.time()
		elapsed = end - start
		times.loc[i, script] = elapsed
		#print(result.stdout)
		print(f"Time: {elapsed:.2f} seconds")

absolute_end_time=time.time()
times.to_csv("test_graphs_from_all_teams.csv")
print(f"Elapsed time: {absolute_end_time-absolute_start_time:.2} sec")
times_without_graph_names = times.drop(columns=['graph'],inplace=False)
average_time = times_without_graph_names.mean(axis=1)
times['average_graph_time'] = average_time
times = times.sort_values(by='average_graph_time', ascending=False).reset_index(drop=True)
times.to_csv("test_graphs_from_all_teams_sorted.csv")
#test_times - prvi pogon vseh testov, kjer so nekatere implementacije še vedno napačno delale
#test_times2 - popravljene implementacije, ki niso delale, sprememba grafov, da se dajo stisniti v 50MB
#test_times3 - sprememba zadnjega grafa (9600), tako da ima k = 42
#test_times4 - sprememba grafov, da imajo t na (-10000, -10000), (če to vpliva na implementacije z A* search?)
