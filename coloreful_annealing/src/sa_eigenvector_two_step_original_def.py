#!/bin/env python3
# pylint: disable=C0111 # missing function docstring
# pylint: disable=C0103 # UPPER case


import random, numpy as np
import networkx as nx
from annealer import Annealer

class SymmetryAproximator(Annealer):
    def __init__(self, state, A, B, mfp, probability_constant = 0.1, division_constant = 0.1):
        """state - initial permutation, A - adjacency matrix of a graph, B - in this case just A, mfp - maximum fixed points, probability_constant, division_constant - constants used when working with similarities"""
        self.N, self.mfp, self.lfp, self.fp = A.shape[0], mfp, 0, 0
        self.A = self.B = A
        self.division_constant = division_constant
        self.probability_constant = probability_constant
        if B is not None:
            self.B = B
        self.iNeighbor, self.dNeighbor = [], [set() for _ in range(self.N)]
        for i in range(self.N):
            neigh = set()
            for j in range(self.N):
                if A[j,i] == 1 and i != j:
                    neigh.add(j)
            self.iNeighbor.append(neigh)
                    

        # compute the similarity matrix. If two vertices have similar degree, they will have a higher value in this matrix
        self.similarity_matrix = self.compute_similarity_matrix(self.A, division_constant=self.division_constant)
        
        
        for i, s in enumerate(state):
            neigh = []
            for j in range(self.N):
                if self.B[j,i] == 1:
                    neigh.append(j)
                    self.dNeighbor[s].add(state[j])
        
        super(SymmetryAproximator, self).__init__(state) 

    # compute dE for vertex swap
    def diff(self, a, aN, b, bN):
        c = len(self.iNeighbor[a].symmetric_difference(aN))
        d = len(self.iNeighbor[b].symmetric_difference(bN))
        return c+d
    
    

    def compute_similarity_matrix(self, A, division_constant):
        """Input: A - adjacency matrix of a graph. Output: a similarity matrix based on eigenvector centrality of the graph
        In practice, the output can be any similarity matrix here."""
        
        G = nx.from_numpy_array(A)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000) # 100 iterations might not be sufficient
        n = len(eigenvector_centrality)
        diff_matrix = np.zeros((n, n))
        
        # Fill the difference matrix with absolute differences of centralities
        for i in range(n):
            for j in range(n):
                diff_matrix[i, j] = abs(eigenvector_centrality[i] - eigenvector_centrality[j])
        
        # compute the inverse of the distance matrix - create a form of similariy measure.
        # Add a constant to avoid division by zero. The higher the constant, the more even the choices will be
        similarity_matrix = 1./(division_constant + diff_matrix)
        
        return similarity_matrix
        

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
        a = random.randint(0, len(self.state) - 1) #random choice works significantly better then a choice based on how "bad" the vertex and its image are
        
        
        # choose the vertex to swap images with:


        # list of energy differences for each possible swap. Start with zeros
        energy_diffs = np.zeros(len(self.state))
        image_a = self.state[a] 
        
        sim_a = self.similarity_matrix.item(a,image_a) # similarity of a -> image of a
        
        for b in range(len(self.state)):
            image_b = self.state[b]

            # if the swap would create a fixed point, let the energy difference be 0 (it will be normalized later)
            if (not (a == image_b or b == image_a or a == b)):
                sim_b = self.similarity_matrix.item(b,image_b) # similarity of b -> image of b
            
                sim_a_new = self.similarity_matrix.item(a,image_b) # similarity of a -> image of b
                sim_b_new = self.similarity_matrix.item(b,image_a) # similarity of b -> image of a
           
            
                # Calculate energy difference. We want this value to be as large as possible
                energy_diff = sim_a_new + sim_b_new - sim_a - sim_b
                energy_diffs[b] = energy_diff
                
        # choose the second vertex b that will swap images with a using the energy differences as a probability distribution.
        # probability constant is used to avoid division by zero. Also, the higher it is, the more even the choices will be
        energy_diffs = [max(self.probability_constant, energy_diff) for energy_diff in energy_diffs]
        # Normalize to create a probability distribution
        energy_diffs_probs = np.array(energy_diffs) / sum(energy_diffs)
        
        # Choose b based on energy differences as probability distribution
        b = np.random.choice(range(len(self.state)), p = energy_diffs_probs)
            
        if self.check_fp(a,b):
            ida, idb = self.state[a], self.state[b]
            aN, bN = self.dNeighbor[ida], self.dNeighbor[idb]
            
            # compute initial energy
            initial = self.diff(ida, aN, idb, bN) 
            self.rewire(a,b,False)
            # update permutation
            self.state[a], self.state[b] = self.state[b], self.state[a]
            aN, bN = self.dNeighbor[ida], self.dNeighbor[idb]
            
            # compute new energy
            after = self.diff(ida, aN, idb, bN)
            return (after-initial)/2, a, b
        return 0, a, b

def check(perm):
    """Check if a permutation has a fixed point"""
    for i, v in enumerate(perm):
        if i == v:
            return True
    return False

def annealing(a, b=None, temp=1, steps=30000, runs=1, fp=0, division_constant=0.1, probability_constant=0.1):
    best_state, best_energy = None, None
    N = len(a)
    for _ in range(runs): 
        perm = np.random.permutation(N)
        while check(perm):
            perm = np.random.permutation(N)
            
        SA = SymmetryAproximator(list(perm), a, b, fp, division_constant=division_constant, probability_constant=probability_constant)
        SA.Tmax = temp
        SA.Tmin = 0.01
        SA.steps = steps
        SA.copy_strategy = 'slice'
        state, e = SA.anneal()
        e = 4 * e / (N*(N-1))
        if best_energy == None or e < best_energy:
            best_state, best_energy = state, e

    return best_state, best_energy
