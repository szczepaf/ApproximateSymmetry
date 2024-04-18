import networkx as nx
import new_sa as sa # import the given algorithm that we want to test
import datetime
import csv
import ast

def get_ER_graphs(simulation_count = 30, edge_densities = [0.1, 0.3, 0.5], sizes = [20, 50, 100]):
    graphs = []
    for i in range(simulation_count):
        for size in sizes:
            for edge_density in edge_densities:
                graph = nx.erdos_renyi_graph(size, edge_density)
                # store the graph and desired density
                graphs.append((graph, edge_density, "ER"))
    return graphs

def get_grids(simulation_count = 30, vertex_counts = [60, 100]):
    graphs = []
    # 2D graphs
    for i in range(simulation_count):
        for vertex_count in vertex_counts:
            length = 5
            width = vertex_count // length
            graph = nx.grid_graph(dim = (length, width))
            
            graphs.append((graph, 2, "grid"))
           
    # 3D graphs
    for i in range(simulation_count):
        for vertex_count in vertex_counts:
            length = 5
            width = 2
            heigth = vertex_count // (length * width)
            graph = nx.grid_graph(dim = (length, width, heigth))
            
            graphs.append((graph, 3, "grid"))
            
    return graphs

def get_barabasi_albert_graphs(simulation_count = 30, k = [3, 6], sizes = [60, 100]):
    graphs = []
    for i in range(simulation_count):
        for size in sizes:
            for k_value in k:
                graph = nx.barabasi_albert_graph(size, k_value)
                # store the graph and desired density
                graphs.append((graph, k_value, "BA"))
    return graphs

def get_stochastic_block_model_graphs(simulation_count = 30, sizes = [60, 120], block_counts = [2, 4], p_in = 0.5, p_out = 0.1):
    graphs = []
    for i in range(simulation_count):
        for size in sizes:
            for block_count in block_counts:
                graph = nx.stochastic_block_model([size // block_count] * block_count, [[p_in if i == j else p_out for i in range(block_count)] for j in range(block_count)])
                # store the graph and desired density
                graphs.append((graph, block_count, "SBM"))
    return graphs

def run_simulation_on_parameters(graph_tuples, steps, temp_max, temp_min, filename, alg_type):
    for graph_tuple in graph_tuples:
        print("Processing graph: ", graph_tuples.index(graph_tuple) + 1, " out of ", len(graph_tuples), " (in current epoch)." )
        graph = graph_tuple[0]
        parameter = graph_tuple[1]
        graph_type = graph_tuple[2]
        
        perm, S = sa.annealing(nx.to_numpy_array(graph), steps=steps, temp=temp_max, temp_min=temp_min)

        vertex_count = graph.number_of_nodes()
        
        fp = 0
        for i in range(len(perm)):
            if perm[i] == i:
                fp += 1
                
        
        row = [alg_type, graph_type, vertex_count, parameter, steps, temp_max, temp_min, S, fp]
        
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
            

def load_graphs_from_csv(filename):
    graph_tuple_list = []
    with open(filename, 'r', newline='') as csvfile:
        graph_reader = csv.reader(csvfile)
        for row in graph_reader:
            graph_type, param, edge_list_str = row
            # Use ast.literal_eval to safely evaluate the string representation of the edge list
            edge_list = ast.literal_eval(edge_list_str)
            # Create a new graph from the edge list
            graph = nx.parse_edgelist(edge_list, nodetype=int)
            # Correctly interpret the parameter as an integer or a float if necessary
            param = float(param) if '.' in param else int(param)
            graph_tuple_list.append((graph, param, graph_type))
    return graph_tuple_list


def main():
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    algorithm_type = "sa_temperature"
    steps = 50000
    # original values for grid search - phase 1. Best: 0.1x0.01
    #temp_max = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    #temp_min = [0.0001, 0.001, 0.01, 0.1, 1]

    # phase 2 - TO BE RUN?
    temp_max = [0.05]
    temp_min = [0.001, 0.01, 0.1, 1]

    graph_tuples = load_graphs_from_csv("graphs_for_grid_search/graphs_for_grid_search_26_03.csv")   # These are the used graphs  
    file_name = f"grid_search_results/symmetry_results_for_temperature_remaining_{date}_{algorithm_type}.csv"
    
    #write header row
    with open(file_name, "w") as f:
        f.write("alg_type,graph_type,vertex_count,parameter,steps,temp_max,temp_min,energy,fps\n")
     

    
    i, total_epochs = 0, len(temp_max) * len(temp_min)

    for t in temp_max:
        for s in temp_min:
            print("Epoch: ", i + 1, " out of ", total_epochs)

            if (temp_min > temp_max):
                i+=1
                continue
            i += 1
            run_simulation_on_parameters(graph_tuples, steps, t, s, file_name, algorithm_type)
    
    print("Finished.")   


if __name__ == "__main__":
    main()
