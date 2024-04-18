#!/bin/env python3
# pylint: disable=C0111 # missing function docstring
# pylint: disable=C0103 # UPPER case


from importlib.metadata import packages_distributions
import math
import random, numpy as np
from typing_extensions import override
import time
import networkx as nx
from annealer import Annealer 


def count_fp(perm):
    count = 0
    for i, v in enumerate(perm):
        if i == v:
            count += 1
    return count

class SymmetryAproximator(Annealer):
    def __init__(self, state, A, B, mfp = float(math.inf), probability_constant = 0.2, division_constant = 0.1):
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
    
    
    
    @override
    def anneal(self):
        """Minimizes the energy of a system by simulated annealing.

        Parameters
        state : an initial arrangement of the system

        Returns
        (state, energy): the best state and energy found.
        """
        step = 0
        self.start = time.time()

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
        Tfactor = -math.log(self.Tmax / self.Tmin)

        # Note initial state
        T = self.Tmax
        E = self.energy()
        prevState = self.copy_state(self.state)
        prevEnergy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        trials, accepts, improves = 0, 0, 0
        # NO UPDATES
        # if self.updates > 0:
        #     updateWavelength = self.steps / self.updates
        #     self.update(step, T, E, None, None)

        # Attempt moves to new states
        while step < self.steps and not self.user_exit:
            step += 1
            # T = self.Tmax * math.exp(Tfactor * step / self.steps)
            T = self.Tmax / (np.log(5+step))
            dE, a, b = self.move()
            
            
            if dE is None:
                E = self.energy()
                dE = E - prevEnergy
            else:
                E += dE
            trials += 1
            
            # Work with the new definition of Symmetry taking into account the number of fixed points. 
            E = E * (self.N * (self.N - 1) - self.lfp * (self.lfp - 1)) / (self.N * (self.N - 1) - self.fp * (self.fp - 1))
            
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                # Restore previous state
                self.rewire(a,b, True)
                self.state = self.copy_state(prevState)
                E = prevEnergy
            else:
                # Accept new state and compare to best state
                accepts += 1
                if dE < 0.0:
                    improves += 1
                prevState = self.copy_state(self.state)
                prevEnergy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E

            # NO UPDATES
            # if self.updates > 1:
            #     if (step // updateWavelength) > ((step - 1) // updateWavelength):
            #         self.update(
            #             step, T, E, accepts / trials, improves / trials)
            #         trials, accepts, improves = 0, 0, 0

        self.state = self.copy_state(self.best_state)

        if self.save_state_on_exit:
            self.save_state()

        # Return best state and energy
        return self.best_state, self.best_energy
    

    def compute_similarity_matrix(self, A, division_constant):
        """Input: A - adjacency matrix of a graph. Output: a similarity matrix based on betweenness centrality of the graph
        In practice, the output can be any similarity matrix here."""
        
        # create the graph from the adjacency matrix
        G = nx.from_numpy_array(A)
        #compute the centrality
        betweenness_centrality = nx.betweenness_centrality(G)
        # Initialize the difference matrix with zeros
        n = len(betweenness_centrality)
        diff_matrix = np.zeros((n, n))
        # Fill the difference matrix with absolute differences of centralities
        for i in range(n):
            for j in range(n):
                diff_matrix[i, j] = abs(betweenness_centrality[i] - betweenness_centrality[j])
        
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
        return diff/ 4
    

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
        #if temp > self.mfp: will not happen in new S(A) definition
        #    return False 
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


def annealing(a, b=None, temp=1, steps=30000, runs=1, fp=float(math.inf), division_constant=0.1, probability_constant=0.2):
    best_state, best_energy = None, None
    N = len(a)
    for _ in range(runs): 
        perm = np.random.permutation(N)
        while check(perm):
            perm = np.random.permutation(N)
            

        # It has to be list(perm), not just perm! To actually make a copy that will be modified by the algorithm
        SA = SymmetryAproximator(list(perm), a, b, fp, division_constant=division_constant, probability_constant=probability_constant)
        SA.Tmax = temp
        SA.Tmin = 0.01
        SA.steps = steps
        SA.copy_strategy = 'slice'
        state, _ = SA.anneal()
        
        fps_in_best_state = count_fp(state)
        
        e = 4 * SA.energy() / (( N * ( N - 1 )) - (fps_in_best_state * (fps_in_best_state - 1)))
        if best_energy == None or e < best_energy:
            best_state, best_energy = state, e
            
    return best_state, best_energy 




def count_fp(perm):
    count = 0
    for i, v in enumerate(perm):
        if i == v:
            count += 1
    return count
