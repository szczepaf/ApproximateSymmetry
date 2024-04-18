import numpy as np
import networkx as nx
import new_sa as sa
import sa_pagerank_twostep as sa_improved

import matplotlib.pyplot as plt





def get_duplication_divergence_graphs(simulation_count = 50, vertex_counts = [300, 500], probabilities=[0.05, 0.1, 0.3]):
    graphs = []
    for i in range(simulation_count):
        for vertex_count in vertex_counts:
            for probability in probabilities:
                graph = nx.duplication_divergence_graph(vertex_count, probability)
                graphs.append((graph, probability))
    return graphs

        

            

         
            

def run_simulation_on_DD(graphs, steps):
    # write header row into the results csv
    graph_type = "DD"
    file_name = f"comparisons/comparisons_of_pagerank_vs_og_on_large_DD.csv"
    with open(file_name, "w") as f:
        f.write("type,vertex_count,probability,pagerank_fp,pagerank_energy,og_fp,og_energy\n")

    
    for graph_tuple in graphs:
        graph = graph_tuple[0]
        probability = graph_tuple[1]
        
        
        
        print("Processing graph: ", graphs.index(graph_tuple) + 1, " out of ", len(graphs), " in current epoch." )

        vertex_count = graph.number_of_nodes()
 
        nperm, nS = sa_improved.annealing(nx.to_numpy_array(graph), steps=steps)
        og_perm, og_s = sa.annealing(nx.to_numpy_array(graph), steps=steps)
        
        
        #count fixed points in the permutations
        nfp, fp_og = 0, 0
        for i in range(vertex_count):
            if nperm[i] == i:
                nfp += 1
            if og_perm[i] == i:
                fp_og += 1

        
        # store the results in a csv file
        with open(file_name, "a") as f:
            f.write(f"{graph_type},{vertex_count},{probability},{nfp},{nS},{fp_og},{og_s}\n")
        


def main():
    # Individual graph type symmetries
    DD_graphs = get_duplication_divergence_graphs()
    
    steps = 30000
    print("DD")
    run_simulation_on_DD(DD_graphs, steps)
    
    print("Done.")

if __name__ == "__main__":
    main()