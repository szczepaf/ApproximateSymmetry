import numpy as np
import networkx as nx
import new_sa as sa
import sa_eigenvector_two_step as sa_improved

import matplotlib.pyplot as plt





def get_large_barabasi_albert_graphs(simulation_count = 50, k = [10, 15, 20], sizes = [300, 500]):
    graphs = []
    for i in range(simulation_count):
        for size in sizes:
            for k_value in k:
                graph = nx.barabasi_albert_graph(size, k_value)
                # store the graph and desired density
                graphs.append((graph, k_value))
    return graphs

        

            

         
            

def run_simulation_on_BA(graphs, steps):
    # write header row into the results csv
    graph_type = "BA"
    file_name = f"comparisons/comparisons_of_eigenvector_vs_og_on_large_BA.csv"
    with open(file_name, "w") as f:
        f.write("type,vertex_count,k,eigenvector_fp,eigenvector_energy,og_fp,og_energy\n")

    
    for graph_tuple in graphs:
        graph = graph_tuple[0]
        k = graph_tuple[1]
        
        
        
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
            f.write(f"{graph_type},{vertex_count},{k},{nfp},{nS},{fp_og},{og_s}\n")
        


def main():
    # Individual graph type symmetries
    BA_graphs = get_large_barabasi_albert_graphs()
    
    steps = 30000
    print("BA")
    run_simulation_on_BA(BA_graphs, steps)
    
    print("Done.")

if __name__ == "__main__":
    main()