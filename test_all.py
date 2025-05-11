import subprocess
import time
import pandas as pd

graph_paths =[ 
			  "inst1_comb_k0.txt",
			  "test_graph_1_subgraphs_100.txt",
			  "inst2_block_cheb.txt",
			  "test_graph_1_subgraphs_1024.txt",
			  "inst3_spiral.txt",
			  "inst4_tcomb.txt",
			  "test_graph_1_subgraphs_500.txt",
			  "inst5_flower.txt",
			  "test_graph_2_subgraphs_5000.txt",
			  "test_graph_3_subgraphs_9600.txt",
			  ]

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
print(f"Elapsed time: {absolute_end_time-absolute_start_time:.2} sec")
times.to_csv("test_times_2.csv")
