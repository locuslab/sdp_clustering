import sys
from scipy.io import mmread
import torch
sys.path.append("./")
from cluster import mixing_cluster, mixing_aggregate, SparseMat

fname = 'zachary.mtx'

A = SparseMat.from_scipy(mmread('data/'+fname))
n = len(A.indptr)-1
Adiag = torch.zeros(n)

k = 8
max_outer_iter = 10
max_lv = 10
comm_init = None
inner_iter = 2
eps = 1e-6

for it in range(max_outer_iter):
    comms = []
    G, Gdiag = A.clone(), Adiag.clone()
    for lv in range(max_lv):
        print(f'\nouter iter {it+1} lv {lv+1}\n')
        fval, comm, n_comm, V = mixing_cluster(G, Gdiag, k, comm=comm_init, eps=eps, max_iter=inner_iter, comm_init=comm_init is not None)
        print(fval, n_comm)
        fval, new_comm, new_n_comm, V = mixing_cluster(G, Gdiag, k, comm=comm, n_comm=n_comm, eps=1e-4, max_iter=1, shrink=1)
        print(fval, new_n_comm)

        comm_init = torch.zeros(new_n_comm, dtype=torch.int32)
        for i, ic in enumerate(new_comm.tolist()):
            comm_init[ic] = comm[i]

        comms.append(new_comm.clone().detach_())
        G, Gdiag = mixing_aggregate(G, Gdiag, new_comm, new_n_comm)

    for lv in reversed(range(len(comms)-1)):
        for i, ic in enumerate(comms[lv].tolist()):
            comms[lv][i] = comms[lv+1][ic]
    comm_init = comms[0].clone().detach_()
