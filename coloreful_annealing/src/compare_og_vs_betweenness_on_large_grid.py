import numpy as np
import networkx as nx
import new_sa as sa
import sa_betweenness_twostep as sa_improved

import matplotlib.pyplot as plt





def get_grids(simulation_count = 50, vertex_counts = [300, 500]):
    graphs = []
    # 2D graphs
    for i in range(simulation_count):
        for vertex_count in vertex_counts:
            length = 10
            width = vertex_count // length
            graph = nx.grid_graph(dim = (length, width))
            
            graphs.append((graph, 2))
           
    # 3D graphs
    for i in range(simulation_count):
        for vertex_count in vertex_counts:
            length = 10
            width = 5
            heigth = vertex_count // (length * width)
            graph = nx.grid_graph(dim = (length, width, heigth))
            
            graphs.append((graph, 3))
            
    return graphs
        

            

         
            

def run_simulation_on_grid(graphs, steps):
    # write header row into the results csv
    graph_type = "grid"
    file_name = f"comparisons/comparisons_of_betweenness_vs_og_on_large_grid.csv"
    with open(file_name, "w") as f:
        f.write("type,vertex_count,dimension,betweenness_fp,betweenness_energy,og_fp,og_energy\n")

    
    for graph_tuple in graphs:
        graph = graph_tuple[0]
        dimension = graph_tuple[1]
        
        
        
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
            f.write(f"{graph_type},{vertex_count},{dimension},{nfp},{nS},{fp_og},{og_s}\n")
        


def main():
    # Individual graph type symmetries
    grids = get_grids()
    
    steps = 30000
    print("grid")
    run_simulation_on_grid(grids, steps)
    
    print("Done.")

if __name__ == "__main__":
    main()