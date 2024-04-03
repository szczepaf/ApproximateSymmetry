from math import inf
import numpy as np
import networkx as nx
import sa_eigenvector_centrality_two_step as sa_eigen_n # import the improved algorithm
import sa_eigenvector_centrality_two_step_original_def as sa_eigen_og
import sa_degree_pair_likelihood_two_step as deg_sa
import sa as sa
import datetime

steps = 30000
fp = 75

ba_graph = nx.barabasi_albert_graph(150, 3)

a = nx.to_numpy_array(ba_graph)
perm, S = sa_eigen_n.annealing(a, steps = steps, fp = 100000, division_constant=0.1, probability_constant=0.1)
permOG, SOG = sa_eigen_og.annealing(a, steps=steps, fp = fp, division_constant=0.1, probability_constant=0.1)
degperm, degS = deg_sa.annealing(a, steps = steps, fp = 10000, division_constant=0.1, probability_constant=0.1)
sa, SS = sa.annealing(a, steps=steps, fp = fp)

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

adj = nx.to_numpy_array(ba_graph)
e1 = energy(adj, perm)
e2 = energy(adj, permOG)
e3 = energy(adj, degperm)
e4 = energy(adj, sa)

print(S, e1)
print(SOG, e2)
print(degS, e3)
print(SS, e4)

