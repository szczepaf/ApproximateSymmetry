#!/bin/env python3
# pylint: disable=C0111 # missing function docstring
# pylint: disable=C0103 # UPPER case

import random, numpy as np
from annealer import Annealer

class SymmetryAproximator(Annealer):
    def __init__(self, state, A, B, mfp):
        self.N, self.mfp, self.lfp, self.fp = A.shape[0], mfp, 0, 0
        self.A = self.B = A
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
        if temp > self.mfp:
            return False
        self.lfp = self.fp
        self.fp = temp
        return True
        

    def move(self):
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

def annealing(a, b=None, temp=1, temp_min = 0.01, steps=20000, runs=1, fp=0):
    best_state, best_energy = None, None
    N = len(a)
    for _ in range(runs): 
        perm = np.random.permutation(N)
        # only permutations with fixed point
        while check(perm):
            perm = np.random.permutation(N)
        SA = SymmetryAproximator(list(perm), a, b, fp)
        SA.Tmax = temp
        SA.Tmin = temp_min  
        SA.steps = steps
        SA.copy_strategy = 'slice'
        state, e = SA.anneal()
        e = 4 * e / ((N*(N-1)))
        if best_energy == None or e < best_energy:
            best_state, best_energy = state, e

    return best_state, best_energy
