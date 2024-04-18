import numpy as np
import networkx as nx
import new_sa as sa
import sa_eigenvector_two_step as sa_eigenvector
import sa_clustering_twostep as sa_clustering
import sa_betweenness_twostep as sa_betweenness
import sa_degree_twostep as sa_degree
import sa_pagerank_twostep as sa_pagerank
import matplotlib.pyplot as plt
import sa_gradient_descend as sa_gradient_descend

def get_duplication_divergence_graphs(simulation_count = 10, vertex_counts = [20, 40], probabilities=[0.1, 0.3]):
    graphs = []
    for i in range(simulation_count):
        for vertex_count in vertex_counts:
            for probability in probabilities:
                graph = nx.duplication_divergence_graph(vertex_count, probability)
                graphs.append((graph, probability))
    return graphs

def get_ER_graphs(simulation_count = 10, edge_densities = [0.1, 0.3], vertex_counts = [40]):
    graphs = []
    for i in range(simulation_count):
        for size in vertex_counts:
            for edge_density in edge_densities:
                graph = nx.erdos_renyi_graph(size, edge_density)
                # store the graph and desired density
                graphs.append((graph, edge_density))
    return graphs

def get_grids(simulation_count = 10, vertex_counts = [30, 50]):
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

def get_barabasi_albert_graphs(simulation_count = 10, k = [2, 4], vertex_counts = [40]):
    graphs = []
    for i in range(simulation_count):
        for size in vertex_counts:
            for k_value in k:
                graph = nx.barabasi_albert_graph(size, k_value)
                # store the graph and desired density
                graphs.append((graph, k_value))
    return graphs


def run_simulation_on_duplication_divergence(graphs, steps, file_format):
    graph_type = "DuplicationDivergence"
    file_name = f"{file_format}_{graph_type}.csv"
    with open(file_name, "w") as f:
        f.write("type,vertex_count,probability,original_fp,original_energy,eigenvector_fp,eigenvector_energy,pagerank_fp, pagerank_energy, betweenness_fp, betweenness_energy, gradient_descend_fp, gradient_descend_energy, gradient_descend_fp, gradient_descend_energy\n")

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
        gradient_descend_perm, gradient_descend_energy = sa_gradient_descend.annealing(adj_matrix, steps=steps)



        # Count fixed points in the permutations
        og_fp = sum(1 for i in range(vertex_count) if og_perm[i] == i)
        eigenvector_fp = sum(1 for i in range(vertex_count) if eigenvector_perm[i] == i)
        pagerank_fp = sum(1 for i in range(vertex_count) if pagerank_perm[i] == i)
        betweenness_fp = sum(1 for i in range(vertex_count) if betweenness_perm[i] == i)
        gradient_descent_fp = sum(1 for i in range(vertex_count) if gradient_descend_perm[i] == i)
        


        # Store the results
        with open(file_name, "a") as f:
            f.write(f"{graph_type},{vertex_count},{probability},{og_fp},{og_energy},{eigenvector_fp},{eigenvector_energy},{pagerank_fp},{pagerank_energy},{betweenness_fp},{betweenness_energy},{gradient_descent_fp},{gradient_descend_energy}\n")


           

def run_simulation_on_grid(graphs, steps, file_format):
    # write header row into the results csv
    graph_type = "grid"
    file_name = f"{file_format}_{graph_type}.csv"
    with open(file_name, "w") as f:
        f.write("type,vertex_count,dimension,original_fp,original_energy,eigenvector_fp,eigenvector_energy,pagerank_fp, pagerank_energy, betweenness_fp, betweenness_energy, gradient_descend_fp, gradient_descend_energy\n")

    
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
        gradient_descend_perm, gradient_descend_energy = sa_gradient_descend.annealing(adj_matrix, steps=steps)

        
        #count fixed points in the permutations
        og_fp, eigenvector_fp, pagerank_fp, betweenness_fp, gradient_descent_fp = 0, 0, 0, 0, 0
        for i in range(vertex_count):
            if og_perm[i] == i:
                og_fp += 1
            if eigenvector_perm[i] == i:
                eigenvector_fp += 1
            if pagerank_perm[i] == i:
                pagerank_fp += 1
            if betweenness_perm[i] == i:
                betweenness_fp += 1

        

        
        # store the results in a csv file
        with open(file_name, "a") as f:
            f.write(f"{graph_type},{vertex_count},{dimension},{og_fp},{og_energy},{eigenvector_fp},{eigenvector_energy},{pagerank_fp},{pagerank_energy},{betweenness_fp},{betweenness_energy},{gradient_descent_fp},{gradient_descend_energy}\n")
            

def run_simulation_on_ER(graphs, steps, file_format):
    # write header row into the results csv
    graph_type = "ER"
    file_name = f"{file_format}_{graph_type}.csv"
    with open(file_name, "w") as f:
        f.write("type,vertex_count,edge_density,original_fp,original_energy,eigenvector_fp,eigenvector_energy,pagerank_fp, pagerank_energy, betweenness_fp, betweenness_energy, gradient_descend_fp, gradient_descend_energy\n")

    
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
        gradient_descend_perm, gradient_descend_energy = sa_gradient_descend.annealing(adj_matrix, steps=steps)

        
        #count fixed points in the permutations
        og_fp, eigenvector_fp, pagerank_fp, betweenness_fp, gradient_descent_fp = 0, 0, 0, 0, 0
        for i in range(vertex_count):
            if og_perm[i] == i:
                og_fp += 1
            if eigenvector_perm[i] == i:
                eigenvector_fp += 1
            if pagerank_perm[i] == i:
                pagerank_fp += 1
            if betweenness_perm[i] == i:
                betweenness_fp += 1
            

        # store the results in a csv file
        with open(file_name, "a") as f:
            f.write(f"{graph_type},{vertex_count},{edge_density},{og_fp},{og_energy},{eigenvector_fp},{eigenvector_energy},{pagerank_fp},{pagerank_energy},{betweenness_fp},{betweenness_energy},{gradient_descent_fp},{gradient_descend_energy}\n")
            

def run_simulation_on_BA(graphs, steps, file_format):
    # write header row into the results csv
    graph_type = "BA"
    file_name = f"{file_format}_{graph_type}.csv"
    with open(file_name, "w") as f:
        f.write("type,vertex_count,k,original_fp,original_energy,eigenvector_fp,eigenvector_energy,pagerank_fp, pagerank_energy, betweenness_fp, betweenness_energy, gradient_descend_fp, gradient_descend_energy\n")

    
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
        gradient_descend_perm, gradient_descend_energy = sa_gradient_descend.annealing(adj_matrix, steps=steps)

        
        #count fixed points in the permutations
        og_fp, eigenvector_fp, pagerank_fp, betweenness_fp, gradient_descent_fp = 0, 0, 0, 0, 0
        for i in range(vertex_count):
            if og_perm[i] == i:
                og_fp += 1
            if eigenvector_perm[i] == i:
                eigenvector_fp += 1
            if pagerank_perm[i] == i:
                pagerank_fp += 1
            if betweenness_perm[i] == i:
                betweenness_fp += 1


        
        # store the results in a csv file
        with open(file_name, "a") as f:
            f.write(f"{graph_type},{vertex_count},{k},{og_fp},{og_energy},{eigenvector_fp},{eigenvector_energy},{pagerank_fp},{pagerank_energy},{betweenness_fp},{betweenness_energy},{gradient_descent_fp},{gradient_descend_energy}\n")
        


def main():
    
    steps = 30000
    simmulation_count = 30
    vertex_sizes = [40]
    
    # Individual graph type symmetries
    grid_graphs = get_grids(simulation_count=simmulation_count, vertex_counts=vertex_sizes)
    ER_graphs = get_ER_graphs(simulation_count=simmulation_count,vertex_counts=vertex_sizes)
    BA_graphs = get_barabasi_albert_graphs(simulation_count=simmulation_count,vertex_counts=vertex_sizes)
    duplication_divergence_graphs = get_duplication_divergence_graphs(simulation_count=simmulation_count,vertex_counts=vertex_sizes)
    
    file_format = "comparisons/comparison_of_centralities_vs_gradient_descend"
    

    #print("DD")
    #run_simulation_on_duplication_divergence(duplication_divergence_graphs, steps, file_format)
    #print("BA")
    #run_simulation_on_BA(BA_graphs, steps, file_format)
    print("ER")
    run_simulation_on_ER(ER_graphs, steps, file_format)
    #print("Grid")
    #run_simulation_on_grid(grid_graphs, steps, file_format)

    
    print("Done.")

if __name__ == "__main__":
    main()