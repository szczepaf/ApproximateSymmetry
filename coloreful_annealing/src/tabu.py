#!/bin/env python3
# pylint: disable=C0111 # missing function docstring
# pylint: disable=C0103 # UPPER case

import math
import random, numpy as np
import time
from typing_extensions import override
from annealer import Annealer

class SymmetryAproximator(Annealer):
    def __init__(self, state, A, B, mfp):
        self.N, self.mfp, self.lfp, self.fp = A.shape[0], mfp, 0, 0
        self.A = self.B = A
        if B is not None:
            self.B = B
                
        # initialize a set of visited permutations
        self.visited = set()
        self.rejected = 0
        

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
        super(SymmetryAproximator, self).__init__(state)  # important!

    # compute dE for vertex swap
    def diff(self, a, aN, b, bN):
        c = len(self.iNeighbor[a].symmetric_difference(aN))
        d = len(self.iNeighbor[b].symmetric_difference(bN))
        return c + d


    # compute 1/4 ||A - pAp^T|| for given p, A
    def energy(self):
        n, m = self.A.shape[0], self.A.shape[1]
        B = self.B[:, self.state]
        diff = 0
        for r in range(n):
            v = self.state[r]
            for c in range(m):
                diff += abs(B[v,c] - self.A[r,c])
                
        energy = diff / 4

        return energy

    

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
        #if temp > self.mfp: # This will not occur in this version
        #    return False
        self.lfp = self.fp
        self.fp = temp
        return True
        

    def move(self):
        # With tabu measures
        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        
        # check whether the new permutation that would emerge by swapping a and b is already visited
        new_state = self.copy_state(self.state)
        new_state[a], new_state[b] = new_state[b], new_state[a]
        if tuple(new_state) in self.visited:
            self.rejected += 1
            return 0, a, b
        
        self.visited.add(tuple(new_state))
        
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

# generate permutation without a fixed point
def check(perm):
    for i, v in enumerate(perm):
        if i == v:
            return True
    return False

def annealing(a, b=None, temp=1, steps=30000, runs=1, fp=float(math.inf)):
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
        fps_in_best_state = count_fp(state)
        e = SA.energy()
        e = 4 * e / (( N * ( N - 1 )) - fps_in_best_state * (fps_in_best_state - 1))
        
        if best_energy == None or e < best_energy:
            best_state, best_energy = state, e
            
    print("rejections: ", SA.rejected)
    return best_state, best_energy 



def count_fp(perm):
    count = 0
    for i, v in enumerate(perm):
        if i == v:
            count += 1
    return count
