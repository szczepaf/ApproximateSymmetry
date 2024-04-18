from re import A
import networkx as nx
import sa_initial_perm as sa_initial_perm
import sa_eigenvector_two_step_initial_perm as sa_eigenvector_initial_perm
import sa_pagerank_two_step_initial_perm as sa_pagerank_initial_perm
import sa_betweenness_two_step_initial_perm as sa_betweenness_initial_perm
import matplotlib as plt
import random



def generate_lrm_graph(n, p, q, k):
    """ Generate a LRM graph with n * 2 nodes, edge probability p, edge acrros parts with probability q and k rewires."""
    G = nx.erdos_renyi_graph(n, p)
    
    G_prime = G.copy()
    mapping = {node: node + n for node in G_prime.nodes()}
    G_prime = nx.relabel_nodes(G_prime, mapping)
    
    L = nx.disjoint_union(G, G_prime)
    
    for i in range(n):
        if random.random() < q:
            L.add_edge(i, i + n)
    
    for _ in range(k):
        while True:
            edge_to_remove = random.choice(list(L.edges()))
            L.remove_edge(*edge_to_remove)
            
            while True:
                new_edge = random.sample(list(L.nodes()), 2)
                if not L.has_edge(*new_edge):
                    L.add_edge(*new_edge)
                    break
                
            break
    
    return L


# generate a Left-Right permutation for LRM graphs with l random shuffles
def generate_left_right_permutation(n, l):
    perm = list(range(n, 2 * n)) + list(range(n))
    for _ in range(l):
        i, j = random.sample(range(n), 2)
        perm[i], perm[j] = perm[j], perm[i]
        
    return perm


 

def hamming_distance_between_permutations(perm1, perm2):
    distance = 0
    for i in range(len(perm1)):
        if perm1[i] != perm2[i]:
            distance += 1
    return distance


def count_fixed_points(permutation):
    fixed_points = 0
    for i in range(len(permutation)):
        if permutation[i] == i:
            fixed_points += 1
    return fixed_points

def energy(adjacency_matrix, permutation):
    n, m = adjacency_matrix.shape[0], adjacency_matrix.shape[1]
    B = adjacency_matrix[:, permutation]
    fp = count_fixed_points(permutation)
    diff = 0
    for r in range(n):
        v = permutation[r]
        for c in range(m):
            diff += abs(B[v,c] - adjacency_matrix[r,c])
                
    normed_energy = diff / ( n * ( n- 1 ) -  fp * (fp - 1)) 

    return normed_energy


def main():
    n = 100
    p = 0.15
    q = 0.25
    k_values = [0, 13, 50, 130] # number of rewires
    l_values = [0, 5, 50, 100, 170, 280, 500] # number of random shuffles of the initial permutation
    simmulation_count = 9
    perfect_LR_permutation = generate_left_right_permutation(n, 0)
    with open("comparisons/measurement_of_distances_from_LR_permutation.csv", "w") as f:
        f.write("n,p,q,k,l,energy_with_perfect_LR_permutation,hamming_distance_sa,sa_s,hamming_distance_eigenvector,sa_eigenvector_s,hamming_distance_pagerank,sa_pagerank_s,hamming_distance_betweenness,sa_betweenness_s\n")
    epoch_counter = 0
    total_epochs = len(k_values) * len(l_values)
    for k in k_values:
        for l in l_values:
            epoch_counter += 1
            print("Epoch ", epoch_counter + 1, " out of ", total_epochs, " epochs.")
            for i in range(simmulation_count):
                print("Processing graph: ", i + 1, " out of ", simmulation_count, " in current epoch.")
                G = generate_lrm_graph(n, p, q, k) # generate the LRM
                A = nx.to_numpy_array(G) 
                l_shuffled_perm = generate_left_right_permutation(n, l)
                energy_with_perfect_LR_permutation = energy(A, perfect_LR_permutation)
                sa_perm, sa_s = sa_initial_perm.annealing(A,state=l_shuffled_perm)
                sa_eigenvector_perm, sa_eigenvector_s = sa_eigenvector_initial_perm.annealing(A,state=l_shuffled_perm)
                sa_pagerank_perm, sa_pagerank_s = sa_pagerank_initial_perm.annealing(A,state=l_shuffled_perm)
                sa_betweenness_perm, sa_betweenness_s = sa_betweenness_initial_perm.annealing(A,state=l_shuffled_perm)
                
                hamming_distance_between_LR_and_SA = hamming_distance_between_permutations(perfect_LR_permutation, sa_perm)
                hamming_distance_between_LR_and_eigenvector = hamming_distance_between_permutations(perfect_LR_permutation, sa_eigenvector_perm)
                hamming_distance_between_LR_and_pagerank = hamming_distance_between_permutations(perfect_LR_permutation, sa_pagerank_perm)
                hamming_distance_between_LR_and_betweenness = hamming_distance_between_permutations(perfect_LR_permutation, sa_betweenness_perm)
                


                with open("comparisons/measurement_of_distances_from_LR_permutation.csv", "a") as f:
                    f.write(f"{n},{p},{q},{k},{l},{energy_with_perfect_LR_permutation},{hamming_distance_between_LR_and_SA},{sa_s},{hamming_distance_between_LR_and_eigenvector},{sa_eigenvector_s},{hamming_distance_between_LR_and_pagerank},{sa_pagerank_s},{hamming_distance_between_LR_and_betweenness},{sa_betweenness_s}\n")
                
                





if __name__ == "__main__":
    main()