import numpy as np
import networkx as nx
import new_sa as sa
import sa_eigenvector_two_step as sa_eigenvector
import sa_clustering_twostep as sa_clustering
import sa_betweenness_twostep as sa_betweenness
import sa_degree_twostep as sa_degree
import sa_pagerank_twostep as sa_pagerank
import matplotlib.pyplot as plt

def get_duplication_divergence_graphs(simulation_count = 100, vertex_counts = [50, 100, 150], probabilities=[0.1, 0.3]):
    graphs = []
    for i in range(simulation_count):
        for vertex_count in vertex_counts:
            for probability in probabilities:
                graph = nx.duplication_divergence_graph(vertex_count, probability)
                graphs.append((graph, probability))
    return graphs

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

def run_simulation_on_duplication_divergence(graphs, steps, file_format):
    graph_type = "DuplicationDivergence"
    file_name = f"{file_format}_{graph_type}.csv"
    with open(file_name, "w") as f:
        f.write("type,vertex_count,probability,original_fp,original_energy,eigenvector_fp,eigenvector_energy,pagerank_fp, pagerank_energy, betweenness_fp, betweenness_energy, clustering_fp, clustering_energy, degree_fp, degree_energy\n")

    for graph_tuple in graphs:
        graph, probability = graph_tuple
        
        print("Processing graph: ", graphs.index(graph_tuple) + 1, " out of ", len(graphs))

        vertex_count = graph.number_of_nodes()
        adj_matrix = nx.to_numpy_array(graph)

        # Perform annealing using various methods
        og_perm, og_energy = sa.annealing(adj_matrix, steps=steps)
        eigenvector_perm, eigenvector_energy = sa_eigenvector.annealing(adj_matrix, steps=steps)
        pagerank_perm, pagerank_energy = sa_pagerank.annealing(adj_matrix, steps=steps)
        betweenness_perm, betweenness_energy = sa_betweenness.annealing(adj_matrix, steps=steps)
        clustering_perm, clustering_energy = sa_clustering.annealing(adj_matrix, steps=steps)
        degree_perm, degree_energy = sa_degree.annealing(adj_matrix, steps=steps)

        # Count fixed points in the permutations
        og_fp = sum(1 for i in range(vertex_count) if og_perm[i] == i)
        eigenvector_fp = sum(1 for i in range(vertex_count) if eigenvector_perm[i] == i)
        pagerank_fp = sum(1 for i in range(vertex_count) if pagerank_perm[i] == i)
        betweenness_fp = sum(1 for i in range(vertex_count) if betweenness_perm[i] == i)
        clustering_fp = sum(1 for i in range(vertex_count) if clustering_perm[i] == i)
        degree_fp = sum(1 for i in range(vertex_count) if degree_perm[i] == i)

        # Store the results
        with open(file_name, "a") as f:
            f.write(f"{graph_type},{vertex_count},{probability},{og_fp},{og_energy},{eigenvector_fp},{eigenvector_energy},{pagerank_fp},{pagerank_energy},{betweenness_fp},{betweenness_energy},{clustering_fp},{clustering_energy},{degree_fp},{degree_energy}\n")


def run_simulation_on_SBM(graphs, steps, file_format):
    # write header row into the results csv
    graph_type = "SBM"
    file_name = f"{file_format}_{graph_type}.csv"
    with open(file_name, "w") as f:
        f.write("type,vertex_count,block_count,original_fp,original_energy,eigenvector_fp,eigenvector_energy,pagerank_fp, pagerank_energy, betweenness_fp, betweenness_energy, clustering_fp, clustering_energy, degree_fp, degree_energy\n")

    
    for graph_tuple in graphs:
        graph = graph_tuple[0]
        block_count = graph_tuple[1]
        
        
        
        print("Processing graph: ", graphs.index(graph_tuple) + 1, " out of ", len(graphs), " in current epoch." )

        vertex_count = graph.number_of_nodes()
        adj_matrix = nx.to_numpy_array(graph)

 
        og_perm, og_energy = sa.annealing(adj_matrix, steps=steps)
        eigenvector_perm, eigenvector_energy = sa_eigenvector.annealing(adj_matrix, steps=steps)
        pagerank_perm, pagerank_energy = sa_pagerank.annealing(adj_matrix, steps=steps)
        betweenness_perm, betweenness_energy = sa_betweenness.annealing(adj_matrix, steps=steps)
        clustering_perm, clustering_energy = sa_clustering.annealing(adj_matrix, steps=steps)
        degree_perm, degree_energy = sa_degree.annealing(adj_matrix, steps=steps)
        
        #count fixed points in the permutations
        og_fp, eigenvector_fp, pagerank_fp, betweenness_fp, clustering_fp, degree_fp = 0, 0, 0, 0, 0, 0
        for i in range(vertex_count):
            if og_perm[i] == i:
                og_fp += 1
            if eigenvector_perm[i] == i:
                eigenvector_fp += 1
            if pagerank_perm[i] == i:
                pagerank_fp += 1
            if betweenness_perm[i] == i:
                betweenness_fp += 1
            if clustering_perm[i] == i:
                clustering_fp += 1
            if degree_perm[i] == i:
                degree_fp += 1
        
        # store the results in a csv file
        with open(file_name, "a") as f:
            f.write(f"{graph_type},{vertex_count},{block_count},{og_fp},{og_energy},{eigenvector_fp},{eigenvector_energy},{pagerank_fp},{pagerank_energy},{betweenness_fp},{betweenness_energy},{clustering_fp},{clustering_energy},{degree_fp},{degree_energy}\n")
            

def run_simulation_on_grid(graphs, steps, file_format):
    # write header row into the results csv
    graph_type = "grid"
    file_name = f"{file_format}_{graph_type}.csv"
    with open(file_name, "w") as f:
        f.write("type,vertex_count,dimension,original_fp,original_energy,eigenvector_fp,eigenvector_energy,pagerank_fp, pagerank_energy, betweenness_fp, betweenness_energy, clustering_fp, clustering_energy, degree_fp, degree_energy\n")

    
    for graph_tuple in graphs:
        graph = graph_tuple[0]
        dimension = graph_tuple[1]
        
        
        
        print("Processing graph: ", graphs.index(graph_tuple) + 1, " out of ", len(graphs), " in current epoch." )

        vertex_count = graph.number_of_nodes()
        adj_matrix = nx.to_numpy_array(graph)

 
        og_perm, og_energy = sa.annealing(adj_matrix, steps=steps)
        eigenvector_perm, eigenvector_energy = sa_eigenvector.annealing(adj_matrix, steps=steps)
        pagerank_perm, pagerank_energy = sa_pagerank.annealing(adj_matrix, steps=steps)
        betweenness_perm, betweenness_energy = sa_betweenness.annealing(adj_matrix, steps=steps)
        clustering_perm, clustering_energy = sa_clustering.annealing(adj_matrix, steps=steps)
        degree_perm, degree_energy = sa_degree.annealing(adj_matrix, steps=steps)
        
        #count fixed points in the permutations
        og_fp, eigenvector_fp, pagerank_fp, betweenness_fp, clustering_fp, degree_fp = 0, 0, 0, 0, 0, 0
        for i in range(vertex_count):
            if og_perm[i] == i:
                og_fp += 1
            if eigenvector_perm[i] == i:
                eigenvector_fp += 1
            if pagerank_perm[i] == i:
                pagerank_fp += 1
            if betweenness_perm[i] == i:
                betweenness_fp += 1
            if clustering_perm[i] == i:
                clustering_fp += 1
            if degree_perm[i] == i:
                degree_fp += 1
        

        
        # store the results in a csv file
        with open(file_name, "a") as f:
            f.write(f"{graph_type},{vertex_count},{dimension},{og_fp},{og_energy},{eigenvector_fp},{eigenvector_energy},{pagerank_fp},{pagerank_energy},{betweenness_fp},{betweenness_energy},{clustering_fp},{clustering_energy},{degree_fp},{degree_energy}\n")
            

def run_simulation_on_ER(graphs, steps, file_format):
    # write header row into the results csv
    graph_type = "ER"
    file_name = f"{file_format}_{graph_type}.csv"
    with open(file_name, "w") as f:
        f.write("type,vertex_count,edge_density,original_fp,original_energy,eigenvector_fp,eigenvector_energy,pagerank_fp, pagerank_energy, betweenness_fp, betweenness_energy, clustering_fp, clustering_energy, degree_fp, degree_energy\n")

    
    for graph_tuple in graphs:
        graph = graph_tuple[0]
        edge_density = graph_tuple[1]
        
        
        
        print("Processing graph: ", graphs.index(graph_tuple) + 1, " out of ", len(graphs), " in current epoch." )

        vertex_count = graph.number_of_nodes()
        adj_matrix = nx.to_numpy_array(graph)

        og_perm, og_energy = sa.annealing(adj_matrix, steps=steps)
        eigenvector_perm, eigenvector_energy = sa_eigenvector.annealing(adj_matrix, steps=steps)
        pagerank_perm, pagerank_energy = sa_pagerank.annealing(adj_matrix, steps=steps)
        betweenness_perm, betweenness_energy = sa_betweenness.annealing(adj_matrix, steps=steps)
        clustering_perm, clustering_energy = sa_clustering.annealing(adj_matrix, steps=steps)
        degree_perm, degree_energy = sa_degree.annealing(adj_matrix, steps=steps)
        
        #count fixed points in the permutations
        og_fp, eigenvector_fp, pagerank_fp, betweenness_fp, clustering_fp, degree_fp = 0, 0, 0, 0, 0, 0
        for i in range(vertex_count):
            if og_perm[i] == i:
                og_fp += 1
            if eigenvector_perm[i] == i:
                eigenvector_fp += 1
            if pagerank_perm[i] == i:
                pagerank_fp += 1
            if betweenness_perm[i] == i:
                betweenness_fp += 1
            if clustering_perm[i] == i:
                clustering_fp += 1
            if degree_perm[i] == i:
                degree_fp += 1

        # store the results in a csv file
        with open(file_name, "a") as f:
            f.write(f"{graph_type},{vertex_count},{edge_density},{og_fp},{og_energy},{eigenvector_fp},{eigenvector_energy},{pagerank_fp},{pagerank_energy},{betweenness_fp},{betweenness_energy},{clustering_fp},{clustering_energy},{degree_fp},{degree_energy}\n")
            

def run_simulation_on_BA(graphs, steps, file_format):
    # write header row into the results csv
    graph_type = "BA"
    file_name = f"{file_format}_{graph_type}.csv"
    with open(file_name, "w") as f:
        f.write("type,vertex_count,k,original_fp,original_energy,eigenvector_fp,eigenvector_energy,pagerank_fp, pagerank_energy, betweenness_fp, betweenness_energy, clustering_fp, clustering_energy, degree_fp, degree_energy\n")

    
    for graph_tuple in graphs:
        graph = graph_tuple[0]
        k = graph_tuple[1]
        
        
        
        print("Processing graph: ", graphs.index(graph_tuple) + 1, " out of ", len(graphs), " in current epoch." )

        vertex_count = graph.number_of_nodes()
        adj_matrix = nx.to_numpy_array(graph)

 
        og_perm, og_energy = sa.annealing(adj_matrix, steps=steps)
        eigenvector_perm, eigenvector_energy = sa_eigenvector.annealing(adj_matrix, steps=steps)
        pagerank_perm, pagerank_energy = sa_pagerank.annealing(adj_matrix, steps=steps)
        betweenness_perm, betweenness_energy = sa_betweenness.annealing(adj_matrix, steps=steps)
        clustering_perm, clustering_energy = sa_clustering.annealing(adj_matrix, steps=steps)
        degree_perm, degree_energy = sa_degree.annealing(adj_matrix, steps=steps)
        
        #count fixed points in the permutations
        og_fp, eigenvector_fp, pagerank_fp, betweenness_fp, clustering_fp, degree_fp = 0, 0, 0, 0, 0, 0
        for i in range(vertex_count):
            if og_perm[i] == i:
                og_fp += 1
            if eigenvector_perm[i] == i:
                eigenvector_fp += 1
            if pagerank_perm[i] == i:
                pagerank_fp += 1
            if betweenness_perm[i] == i:
                betweenness_fp += 1
            if clustering_perm[i] == i:
                clustering_fp += 1
            if degree_perm[i] == i:
                degree_fp += 1

        
        # store the results in a csv file
        with open(file_name, "a") as f:
            f.write(f"{graph_type},{vertex_count},{k},{og_fp},{og_energy},{eigenvector_fp},{eigenvector_energy},{pagerank_fp},{pagerank_energy},{betweenness_fp},{betweenness_energy},{clustering_fp},{clustering_energy},{degree_fp},{degree_energy}\n")
        


def main():
    
    steps = 30000
    simmulation_count = 50
    
    # Individual graph type symmetries
    SBM_graphs = get_stochastic_block_model_graphs(simulation_count=simmulation_count)
    grid_graphs = get_grids(simulation_count=simmulation_count)
    ER_graphs = get_ER_graphs(simulation_count=simmulation_count)
    BA_graphs = get_barabasi_albert_graphs(simulation_count=simmulation_count)
    duplication_divergence_graphs = get_duplication_divergence_graphs(simulation_count=simmulation_count)
    
    file_format = "comparisons/comparison_of_centralities_2"
    

    print("DD")
    run_simulation_on_duplication_divergence(duplication_divergence_graphs, steps, file_format)
    # print("BA")
    # run_simulation_on_BA(BA_graphs, steps, file_format)
    # print("ER")
    # run_simulation_on_ER(ER_graphs, steps, file_format)
    # print("Grid")
    # run_simulation_on_grid(grid_graphs, steps, file_format)
    # print("SBM")
    # run_simulation_on_SBM(SBM_graphs, steps, file_format)
    
    print("Done.")

if __name__ == "__main__":
    main()