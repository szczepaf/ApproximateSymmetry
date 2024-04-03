import networkx as nx
import csv
import ast  # For safely evaluating the string representation of the edge list


def get_ER_graphs(simulation_count = 30, edge_densities = [0.1, 0.3, 0.5], sizes = [50, 100]):
    graphs = []
    for i in range(simulation_count):
        for size in sizes:
            for edge_density in edge_densities:
                graph = nx.erdos_renyi_graph(size, edge_density)
                # store the graph and desired density
                graphs.append((graph, edge_density, "ER"))
    return graphs

def get_grids(simulation_count = 30, vertex_counts = [60, 120]):
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

def get_stochastic_block_model_graphs(simulation_count = 30, sizes = [60, 100], block_counts = [2, 4], p_in = 0.5, p_out = 0.1):
    graphs = []
    for i in range(simulation_count):
        for size in sizes:
            for block_count in block_counts:
                graph = nx.stochastic_block_model([size // block_count] * block_count, [[p_in if i == j else p_out for i in range(block_count)] for j in range(block_count)])
                # store the graph and desired density
                graphs.append((graph, block_count, "SBM"))
    return graphs


def write_graphs_to_csv(graph_tuple_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        graph_writer = csv.writer(csvfile)
        for graph, param, graph_type in graph_tuple_list:
            # Convert the edge list generator to a list of edge tuples
            edge_list = list(nx.generate_edgelist(graph, data=False))
            # Save the edge list as a string
            graph_writer.writerow([graph_type, param, str(edge_list)])

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
    graphs = get_ER_graphs() + get_barabasi_albert_graphs() + get_stochastic_block_model_graphs()
    write_graphs_to_csv(graphs, 'graphs_for_grid_search.csv')
    print("Graphs dumped successfully.")

    
    
if __name__ == '__main__':
    main()