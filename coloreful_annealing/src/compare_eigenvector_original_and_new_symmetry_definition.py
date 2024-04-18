import numpy as np
import networkx as nx
import sa_eigenvector_two_step as sa_eigenvector_new
import sa_eigenvector_two_step_original_def as sa_eigenvector_og
import matplotlib.pyplot as plt


def energy(adjacency_matrix, permutation):
    n, m = adjacency_matrix.shape[0], adjacency_matrix.shape[1]
    B = adjacency_matrix[:, permutation]
    diff = 0
    for r in range(n):
        v = permutation[r]
        for c in range(m):
            diff += abs(B[v,c] - adjacency_matrix[r,c])
                
    normed_energy = diff / (n * (n-1)) # The original defition of S(A)

    return normed_energy

def get_ER_graphs(simulation_count = 100, edge_densities = [0.1, 0.2, 0.3, 0.4, 0.5], sizes = [20, 50, 100]):
    graphs = []
    for i in range(simulation_count):
        for size in sizes:
            for edge_density in edge_densities:
                graph = nx.erdos_renyi_graph(size, edge_density)
                # store the graph and desired density
                graphs.append((graph, edge_density))
    return graphs

def get_grids(simulation_count = 100, vertex_counts = [50, 100, 150]):
    graphs = []
    # 2D graphs
    for i in range(simulation_count):
        for vertex_count in vertex_counts:
            length = 5
            width = vertex_count // length
            graph = nx.grid_graph(dim = (length, width))
            
            graphs.append((graph, 2))
           
    # 3D graphs
    for i in range(simulation_count):
        for vertex_count in vertex_counts:
            length = 5
            width = 2
            heigth = vertex_count // (length * width)
            graph = nx.grid_graph(dim = (length, width, heigth))
            
            graphs.append((graph, 3))
            
    return graphs

def get_barabasi_albert_graphs(simulation_count = 100, k = [3, 5, 7], sizes = [50, 100, 150]):
    graphs = []
    for i in range(simulation_count):
        for size in sizes:
            for k_value in k:
                graph = nx.barabasi_albert_graph(size, k_value)
                # store the graph and desired density
                graphs.append((graph, k_value))
    return graphs

def get_stochastic_block_model_graphs(simulation_count = 100, sizes = [50, 100, 150], block_counts = [2, 3, 5], p_in = 0.5, p_out = 0.1):
    graphs = []
    for i in range(simulation_count):
        for size in sizes:
            for block_count in block_counts:
                graph = nx.stochastic_block_model([size // block_count] * block_count, [[p_in if i == j else p_out for i in range(block_count)] for j in range(block_count)])
                # store the graph and desired density
                graphs.append((graph, block_count))
    return graphs

        

def run_simulation_on_SBM(graphs, steps, file_format):
    # write header row into the results csv
    graph_type = "SBM"
    file_name = f"{file_format}_{graph_type}.csv"
    with open(file_name, "w") as f:
        f.write("type,vertex_count,block_count,new_fp,new_energy,half_fp,half_energy,zero_fp,zero_energy\n")

    
    for graph_tuple in graphs:
        graph = graph_tuple[0]
        block_count = graph_tuple[1]
        
        
        
        print("Processing graph: ", graphs.index(graph_tuple) + 1, " out of ", len(graphs), " in current epoch." )

        vertex_count = graph.number_of_nodes()
        adj_matrix = nx.to_numpy_array(graph)

 
        nperm, nS = sa_eigenvector_new.annealing(adj_matrix, steps=steps)
        perm_half, S_half = sa_eigenvector_og.annealing(adj_matrix, steps=steps , fp = vertex_count // 2)
        perm_zero, S_zero = sa_eigenvector_og.annealing(adj_matrix, steps=steps , fp = 0)
        
        #count fixed points in the permutations
        nfp, fp_half, fp_zero = 0, 0, 0
        for i in range(vertex_count):
            if nperm[i] == i:
                nfp += 1
            if perm_half[i] == i:
                fp_half += 1
            if perm_zero[i] == i:
                fp_zero += 1
        
        nenergy = energy(adj_matrix, nperm)
        energy_half = energy(adj_matrix, perm_half)
        energy_zero = energy(adj_matrix, perm_zero)
        
        # store the results in a csv file
        with open(file_name, "a") as f:
            f.write(f"{graph_type},{vertex_count},{block_count},{nfp},{nenergy},{fp_half},{energy_half},{fp_zero},{energy_zero}\n")
            

def run_simulation_on_grid(graphs, steps, file_format):
    # write header row into the results csv
    graph_type = "grid"
    file_name = f"{file_format}_{graph_type}.csv"
    with open(file_name, "w") as f:
        f.write("type,vertex_count,dimension,new_fp,new_energy,half_fp,half_energy,zero_fp,zero_energy\n")

    
    for graph_tuple in graphs:
        graph = graph_tuple[0]
        dimension = graph_tuple[1]
        
        
        
        print("Processing graph: ", graphs.index(graph_tuple) + 1, " out of ", len(graphs), " in current epoch." )

        vertex_count = graph.number_of_nodes()
        adj_matrix = nx.to_numpy_array(graph)

 
        nperm, nS = sa_eigenvector_new.annealing(adj_matrix, steps=steps)
        perm_half, S_half = sa_eigenvector_og.annealing(adj_matrix, steps=steps , fp = vertex_count // 2)
        perm_zero, S_zero = sa_eigenvector_og.annealing(adj_matrix, steps=steps , fp = 0)
        
        #count fixed points in the permutations
        nfp, fp_half, fp_zero = 0, 0, 0
        for i in range(vertex_count):
            if nperm[i] == i:
                nfp += 1
            if perm_half[i] == i:
                fp_half += 1
            if perm_zero[i] == i:
                fp_zero += 1
        
        nenergy = energy(adj_matrix, nperm)
        energy_half = energy(adj_matrix, perm_half)
        energy_zero = energy(adj_matrix, perm_zero)
        
        # store the results in a csv file
        with open(file_name, "a") as f:
            f.write(f"{graph_type},{vertex_count},{dimension},{nfp},{nenergy},{fp_half},{energy_half},{fp_zero},{energy_zero}\n")
            

def run_simulation_on_ER(graphs, steps, file_format):
    # write header row into the results csv
    graph_type = "ER"
    file_name = f"{file_format}_{graph_type}.csv"
    with open(file_name, "w") as f:
        f.write("type,vertex_count,edge_density,new_fp,new_energy,half_fp,half_energy,zero_fp,zero_energy\n")

    
    for graph_tuple in graphs:
        graph = graph_tuple[0]
        edge_density = graph_tuple[1]
        
        
        
        print("Processing graph: ", graphs.index(graph_tuple) + 1, " out of ", len(graphs), " in current epoch." )

        vertex_count = graph.number_of_nodes()
        adj_matrix = nx.to_numpy_array(graph)

 
        nperm, nS = sa_eigenvector_new.annealing(adj_matrix, steps=steps)
        perm_half, S_half = sa_eigenvector_og.annealing(adj_matrix, steps=steps , fp = vertex_count // 2)
        perm_zero, S_zero = sa_eigenvector_og.annealing(adj_matrix, steps=steps , fp = 0)
        
        #count fixed points in the permutations
        nfp, fp_half, fp_zero = 0, 0, 0
        for i in range(vertex_count):
            if nperm[i] == i:
                nfp += 1
            if perm_half[i] == i:
                fp_half += 1
            if perm_zero[i] == i:
                fp_zero += 1
        
        nenergy = energy(adj_matrix, nperm)
        energy_half = energy(adj_matrix, perm_half)
        energy_zero = energy(adj_matrix, perm_zero)
        
        # store the results in a csv file
        with open(file_name, "a") as f:
            f.write(f"{graph_type},{vertex_count},{edge_density},{nfp},{nenergy},{fp_half},{energy_half},{fp_zero},{energy_zero}\n")
            

def run_simulation_on_BA(graphs, steps, file_format):
    # write header row into the results csv
    graph_type = "BA"
    file_name = f"{file_format}_{graph_type}.csv"
    with open(file_name, "w") as f:
        f.write("type,vertex_count,k,new_fp,new_energy,half_fp,half_energy,zero_fp,zero_energy\n")

    
    for graph_tuple in graphs:
        graph = graph_tuple[0]
        k = graph_tuple[1]
        
        
        
        print("Processing graph: ", graphs.index(graph_tuple) + 1, " out of ", len(graphs), " in current epoch." )

        vertex_count = graph.number_of_nodes()
        adj_matrix = nx.to_numpy_array(graph)

 
        nperm, nS = sa_eigenvector_new.annealing(adj_matrix, steps=steps)
        perm_half, S_half = sa_eigenvector_og.annealing(adj_matrix, steps=steps , fp = vertex_count // 2)
        perm_zero, S_zero = sa_eigenvector_og.annealing(adj_matrix, steps=steps , fp = 0)
        
        #count fixed points in the permutations
        nfp, fp_half, fp_zero = 0, 0, 0
        for i in range(vertex_count):
            if nperm[i] == i:
                nfp += 1
            if perm_half[i] == i:
                fp_half += 1
            if perm_zero[i] == i:
                fp_zero += 1
        
        nenergy = energy(adj_matrix, nperm)
        energy_half = energy(adj_matrix, perm_half)
        energy_zero = energy(adj_matrix, perm_zero)
        
        # store the results in a csv file
        with open(file_name, "a") as f:
            f.write(f"{graph_type},{vertex_count},{k},{nfp},{nenergy},{fp_half},{energy_half},{fp_zero},{energy_zero}\n")
        


def main():
    # Individual graph type symmetries
    SBM_graphs = get_stochastic_block_model_graphs()
    grid_graphs = get_grids()
    ER_graphs = get_ER_graphs()
    BA_graphs = get_barabasi_albert_graphs()
    
    file_format = "comparisons/comparison_of_eigenvector_twostep_in_new_and_original_definition"
    
    steps = 30000
    print("BA")
    run_simulation_on_BA(BA_graphs, steps, file_format)
    print("ER")
    run_simulation_on_ER(ER_graphs, steps, file_format)
    print("Grid")
    run_simulation_on_grid(grid_graphs, steps, file_format)
    print("SBM")
    run_simulation_on_SBM(SBM_graphs, steps, file_format)
    
    print("Done.")

if __name__ == "__main__":
    main()