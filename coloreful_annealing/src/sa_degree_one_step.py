#!/bin/env python3
# pylint: disable=C0111 # missing function docstring
# pylint: disable=C0103 # UPPER case

from importlib.metadata import packages_distributions
import random, numpy as np
from annealer import Annealer

class SymmetryAproximator(Annealer):
    # division constant makes sense even for higher values in this case as the degree distribution has values not only in the range [0,1]
    def __init__(self, state, A, B, mfp, division_constant = 20):
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


        for i, s in enumerate(state):
            neigh = []
            for j in range(self.N):
                if self.B[j,i] == 1:
                    neigh.append(j)
                    self.dNeighbor[s].add(state[j])
                    

        # compute the similarity matrix. If two vertices have similar degree, they will have a higher value here
        self.dist_nodes_matrix_inv = self.compute_similarity_matrix(self.A, division_constant=self.division_constant)
        self.pair_dict = {}
        
        super(SymmetryAproximator, self).__init__(state)  # important!

    # compute dE for vertex swap
    def diff(self, a, aN, b, bN):
        c = len(self.iNeighbor[a].symmetric_difference(aN))
        d = len(self.iNeighbor[b].symmetric_difference(bN))
        return c+d
    

    def compute_similarity_matrix(self, A, division_constant):
        """Input: A - adjacency matrix of a graph. Output: a similarity matrix based on the degree distribution of the graph
        In practice, the output can be any similarity matrix here."""
        
        # sum over columns of the adjacency matrix - get the degree distribution
        degree_distribution = sum(A,0)
        
        # compute a matrix where position (i,j) is the absolute difference between the degree of node i and node j
        dist_nodes_matrix = np.abs(degree_distribution - degree_distribution.reshape(-1,1))
        
        # compute the inverse of the distance matrix - create a form of similariy measure.
        # Add a constant to avoid division by zero. The higher the constant, the more even the choices will be
        dist_nodes_matrix_inv = 1./(division_constant + dist_nodes_matrix)
        

        return dist_nodes_matrix_inv


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
                
        # list of a, image a in the permutation
        images = [(i, self.state[i]) for i in range(len(self.state))]
        move_dict = {}
        # for every pair of elements in the images list, see whether the swap is beneficial
        for i in range(len(images)):
            for j in range(i+1, len(images)):
                

                a, b = images[i][0], images[j][0]
                im_a, im_b = images[i][1], images[j][1]
                ab_sorted = tuple(sorted([a,b]))
                images_sorted = tuple(sorted([im_a,im_b]))
                
                # if the result is already computed, skip the computation
                if (ab_sorted, images_sorted) in self.pair_dict or (images_sorted, ab_sorted) in self.pair_dict:
                    diff = self.pair_dict[(ab_sorted, images_sorted)]
                else:        
                    sim_a = self.dist_nodes_matrix_inv.item(a,im_a)
                    sim_b = self.dist_nodes_matrix_inv.item(b,im_b)
                    sim_a_new = self.dist_nodes_matrix_inv.item(a,im_b)
                    sim_b_new = self.dist_nodes_matrix_inv.item(b,im_a)
                
                    # what would be gained by swapping the images of a and b. If it is a fixed point, let the gain be 0
                    if (a == im_b or b == im_a or a == b):
                        diff = 0
                    else: 
                        diff = sim_a_new + sim_b_new - sim_a - sim_b
                
                    # add the result to the pair dictionary for later use
                
                    self.pair_dict[(ab_sorted, images_sorted)] = diff
                    self.pair_dict[(images_sorted, ab_sorted)] = diff
                    

                    
                
                move_dict[(a,b)] = diff
                
        # choose the best swap with probability. The probability is proportional to the difference in similarity
        # 
        moves = list(move_dict.keys())
        diffs = np.array(list(move_dict.values()))
        # make each diff the max of 0.01 and the diff
        diffs = np.maximum(diffs, 1 / (self.N * 2)) ### THIS IS A PARAMETER THAT CAN BE TUNED
        probs = diffs / np.sum(diffs)
        # Select an index based on diffs as weights
        index = np.random.choice(range(len(moves)), p=probs)
    
        # Retrieve the best swap
        best_swap = moves[index]

        a, b = best_swap
            
        
               


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

def annealing(a, b=None, temp=1, steps=20000, runs=1, fp=0):
    best_state, best_energy = None, None
    N = len(a)
    for _ in range(runs): 
        perm = np.random.permutation(N)
        # only permutations with fixed point
        while check(perm):
            perm = np.random.permutation(N)
        SA = SymmetryAproximator(list(perm), a, b, fp)
        SA.Tmax = temp
        SA.Tmin = 0.01
        SA.steps = steps
        SA.copy_strategy = 'slice'
        state, e = SA.anneal()
        if best_energy == None or e < best_energy:
            best_state, best_energy = state, e
    return best_state, best_energy/(N*(N-1))*4
    return best_state, best_energy
