import torch
import torch.nn as nn
from torch.autograd import Function

import cluster._cpp

class SparseMat(object):
    def __init__(self, indptr, indices, data):
        self.indptr = indptr
        self.indices = indices
        self.data = data

    def clone(self):
        return SparseMat(self.indptr.clone(), self.indices.clone(), self.data.clone())

    @classmethod
    def from_scipy(cls, scipy_mat):
        A = scipy_mat.tocsr()
        indptr = torch.from_numpy(A.indptr)
        indices = torch.from_numpy(A.indices)
        data = torch.from_numpy(A.data).float()

        return cls(indptr, indices, data)

    @classmethod
    def zeros(cls, n, k):
        indptr = torch.arange(0, n*k+1, k, dtype=torch.int32)
        indices = torch.zeros(n*k, dtype=torch.int32)
        data = torch.zeros(n*k)

        return cls(indptr, indices, data)

    @classmethod
    def zeros_like(cls, mat, n=None):
        if n is None: n = mat.indptr.size(0)-1
        indptr = torch.zeros(n+1, dtype=torch.int32)
        indices = torch.zeros_like(mat.indices, dtype=torch.int32)
        data = torch.zeros_like(mat.data)

        return cls(indptr, indices, data)

    def __str__(self):
        n = self.indptr.size(0)-1
        s = []
        for i in range(n):
            s.append(f'({i+1}) ')
            for p in range(self.indptr[i], self.indptr[i+1]):
                s.append(f'{self.indices[p].item()+1}:{self.data[p].item():1.2f}\t')
            s.append('\n')
        return ''.join(s)


def mixing_cluster(A, Adiag, k, comm=None, n_comm=None, max_iter=100, eps=1e-3, shrink=0, comm_init=0):
    n = A.indptr.size(0)-1
    if k>n: k=n
    k_ = max(10, k) # preallocate for increased rank

    V = SparseMat.zeros(n,k)
    buf = torch.zeros(n*k_*2)
    d, s, g = torch.zeros(n), torch.zeros(n*k_), torch.zeros(n*k_)

    queue = torch.zeros(n, dtype=torch.int32)
    is_in = torch.zeros(n, dtype=torch.int32)

    if comm is None: comm = torch.zeros(n, dtype=torch.int32)
    else: comm = comm.clone()
    if n_comm is None: n_comm = torch.zeros(1, dtype=torch.int32)
    else: n_comm = torch.tensor([n_comm], dtype=torch.int32)

    fval = cluster._cpp.cluster(
        max_iter, eps, 
        A.indptr, A.indices, A.data, Adiag,
        V.indptr, V.indices, V.data,
        buf, s, d, g,
        queue, is_in,
        comm, n_comm,
        shrink, comm_init)

    return fval, comm, n_comm.item(), V

def mixing_aggregate(A, Adiag, comm, n_comm):
    G = SparseMat.zeros_like(A, n_comm)
    Gdiag = torch.zeros(n_comm)

    #print(A.indptr.shape, A.indices.shape, A.data.shape, Adiag.shape)
    #print(G.indptr.shape, G.indices.shape, G.data.shape, Gdiag.shape)
    #print(comm.shape)

    cluster._cpp.aggregate(
            A.indptr, A.indices, A.data, Adiag,
            G.indptr, G.indices, G.data, Gdiag,
            comm)

    return G, Gdiag
