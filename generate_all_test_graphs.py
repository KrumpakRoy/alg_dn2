from generate_graph import get_graph

number_of_nodes = [100,500,1000,5000,9600]
k_list = [0,2,-1,4,2]
number_of_subgraphs = [1,1,1,2,3]
for n in number_of_nodes:
	for k in k_list:
		for s in number_of_subgraphs:
			get_graph(n,s,save=True, k=k)
