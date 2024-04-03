#!/bin/env python3
# pylint: disable=C0111 # missing function docstring
# pylint: disable=C0103 # UPPER case

import random, numpy as np
from annealer import Annealer
import networkx as nx

class SymmetryAproximator(Annealer):
    def __init__(self, state, A, B, mfp, modification, division_constant = 0.2):
        self.N, self.mfp, self.lfp, self.fp = A.shape[0], mfp, 0, 0
        self.A = self.B = A
        self.division_constant = division_constant
        if B is not None:
            self.B = B
        self.iNeighbor, self.dNeighbor = [], [set() for _ in range(self.N)]
        for i in range(self.N):
            neigh = set()
            for j in range(self.N):
                if A[j,i] == 1 and i != j:
                    neigh.add(j)
            self.iNeighbor.append(neigh)

        self.modification = modification # my update

        for i, s in enumerate(state):
            neigh = []
            for j in range(self.N):
                if self.B[j,i] == 1:
                    neigh.append(j)
                    self.dNeighbor[s].add(state[j])
                    

        # create the graph from the adjacency matrix
        G = nx.from_numpy_array(A)
        #compute the pagerank
        pagerank = nx.pagerank(G, max_iter = 1000)
        # Initialize the difference matrix with zeros
        n = len(pagerank)
        diff_matrix = np.zeros((n, n))
        # Fill the difference matrix with absolute differences of centralities
        for i in range(n):
            for j in range(n):
                diff_matrix[i, j] = abs(pagerank[i] - pagerank[j])
        
        # compute the inverse of the distance matrix - create a form of similariy measure.
        # Add a constant to avoid division by zero. The higher the constant, the more even the choices will be
        self.similarity_matrix = 1./(division_constant + diff_matrix)
        


        super(SymmetryAproximator, self).__init__(state)  # important!

    # compute dE for vertex swap
    def diff(self, a, aN, b, bN):
        c = len(self.iNeighbor[a].symmetric_difference(aN))
        d = len(self.iNeighbor[b].symmetric_difference(bN))
        return c+d


    # compute 1/4 ||A - pAp^T|| for given p, A
    def energy(self):
        n, m = self.A.shape[0], self.A.shape[1]
        B = self.B[:, self.state]
        diff = 0
        for r in range(n):
            v = self.state[r]
            for c in range(m):
                diff += abs(B[v,c] - self.A[r,c])
        return diff/4
    

    def rewire(self, a, b, reset):
        if reset:
            self.fp = self.lfp
        ida, idb = self.state[a], self.state[b]
        # check whether a, b are neighbors
        neighbors = idb in self.dNeighbor[ida]
        # delete for everyone
        for i in range(self.N):
            if ida in self.dNeighbor[i]:
                self.dNeighbor[i].remove(ida)
            if idb in self.dNeighbor[i]:
                self.dNeighbor[i].remove(idb)

        # add to new neighborhoods
        for n in self.dNeighbor[ida]:
            self.dNeighbor[n].add(idb)
        for n in self.dNeighbor[idb]:
            self.dNeighbor[n].add(ida)

        # fix swapped vertices
        self.dNeighbor[ida], self.dNeighbor[idb] = \
            self.dNeighbor[idb], self.dNeighbor[ida]

        if ida in self.dNeighbor[ida]:
            self.dNeighbor[ida].remove(ida)
        if idb in self.dNeighbor[idb]:
            self.dNeighbor[idb].remove(idb)
        if neighbors:
            self.dNeighbor[ida].add(idb)
            self.dNeighbor[idb].add(ida)

    def check_fp(self, a, b):
        temp = self.fp
        if a == b:
            return False
        if self.state[a] == a:
            temp -= 1
        if self.state[b] == b:
            temp -= 1
        if self.state[a] == b:
            temp += 1
        if self.state[b] == a:
            temp += 1
        if temp > self.mfp:
            return False
        self.lfp = self.fp
        self.fp = temp
        return True
        

    def move(self):
        if self.modification:
            # choose the first vertex randomly
            a = random.randint(0, len(self.state) - 1)
            
            # get the similarity of neighbors of the current node to be chosen, use the similarity matrix computed in the constructor
            dist_nodes = self.similarity_matrix[self.state[a]]
            dist_nodes[self.state[a]] = 0    

            # normalize the similarities for the neighbors into a probability distribution
            #tt = np.random.random(1)
            prbs = np.copy(dist_nodes)/sum(dist_nodes)
            

            #b = np.sum(np.sum(np.sign(tt-np.cumsum(prbs))>0)).astype(int) ### THIS LINE NEEDS TO BE EXPLAINED TO ME
            # try using np.random.choice instead
            b = np.random.choice(range(self.N), p = prbs)

        else: # go with the original random implementation
            a = random.randint(0, len(self.state) - 1)
            b = random.randint(0, len(self.state) - 1)
            
        # enforce trace(P) = 0
        # if self.state[a] != b and self.state[b] != a and a != b:
        if self.check_fp(a,b):
            # compute initial energy
            ida, idb = self.state[a], self.state[b]
            aN, bN = self.dNeighbor[ida], self.dNeighbor[idb]
            initial = self.diff(ida, aN, idb, bN) 
            self.rewire(a,b,False)
            # update permutation
            self.state[a], self.state[b] = self.state[b], self.state[a]
            aN, bN = self.dNeighbor[ida], self.dNeighbor[idb]
            after = self.diff(ida, aN, idb, bN)
            return (after-initial)/2, a, b
        return 0, a, b

# generate permutation without a fixed point
def check(perm):
    for i, v in enumerate(perm):
        if i == v:
            return True
    return False

def annealing(a, b=None, temp=1, steps=20000, runs=1, fp=0, modification = True, division_constant = 1.5):
    best_state, best_energy = None, None
    N = len(a)
    for _ in range(runs): 
        perm = np.random.permutation(N)
        # only permutations with fixed point
        while check(perm):
            perm = np.random.permutation(N)
        SA = SymmetryAproximator(list(perm), a, b, fp, modification, division_constant)
        SA.Tmax = temp
        SA.Tmin = 0.01
        SA.steps = steps
        SA.copy_strategy = 'slice'
        state, e = SA.anneal()
        if best_energy == None or e < best_energy:
            best_state, best_energy = state, e
    return best_state, best_energy/(N*(N-1))*4
    return best_state, best_energy
