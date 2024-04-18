from math import inf
import numpy as np
import networkx as nx
import sa_eigenvector_two_step as sa_eigenvector
import sa_eigenvector_two_step_tabu as sa_eigenvector_tabu
import datetime
import new_sa as sa
import sa_wl as sa_wl
import tabu as tabu
import sa_wl_tabu as wl_tabu
import sa_gradient_descend as sa_gradient_descend
import random
import sa_betweenness_twostep as sa_betweenness
import sa_degree_twostep as sa_degree
import sa_eigenvector_two_step_initial_perm as sa_eigenvector_initial_perm
import sa_initial_perm as sa_initial_perm
import sa_gradient_descend as sa_gradient_descend


steps = 30000
G = nx.erdos_renyi_graph(50, 0.1)
A = nx.to_numpy_array(G)

s, p = sa_gradient_descend.annealing(A, steps = 50)

