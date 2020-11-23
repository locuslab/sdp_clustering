import numpy as np
import sdp_clustering._cpp as _cpp

class SparseMat(object):
    def __init__(self, indptr, indices, data):
        self.indptr = indptr
        self.indices = indices
        self.data = data

    def copy(self):
        return SparseMat(self.indptr.copy(), self.indices.copy(), self.data.copy())

    @classmethod
    def from_scipy(cls, scipy_mat):
        A = scipy_mat.tocsr()
        indptr = A.indptr
        indices = A.indices
        data = np.asarray(A.data, dtype=np.float32)

        return cls(indptr, indices, data)

    @classmethod
    def zeros(cls, n, k):
        indptr = np.arange(0, n*k+1, k, dtype=np.int32)
        indices = np.zeros(n*k, dtype=np.int32)
        data = np.zeros(n*k)

        return cls(indptr, indices, data)

    @classmethod
    def zeros_like(cls, mat, n=None):
        if n is None: n = mat.indptr.shape[0]-1
        indptr = np.zeros(n+1, dtype=np.int32)
        indices = np.zeros_like(mat.indices, dtype=np.int32)
        data = np.zeros_like(mat.data)

        return cls(indptr, indices, data)

    def __str__(self):
        n = self.indptr.shape[0]-1
        s = []
        for i in range(n):
            s.append(f'({i+1}) ')
            for p in range(self.indptr[i], self.indptr[i+1]):
                s.append(f'{self.indices[p].item()+1}:{self.data[p].item():1.2f}\t')
            s.append('\n')
        return ''.join(s)


def mixing_cluster(A, Adiag, k, comm=None, n_comm=None, max_iter=100, eps=1e-3, shrink=0, comm_init=0):
    n = A.indptr.shape[0]-1
    if k>n: k=n
    k_ = max(10, k) # preallocate for increased rank

    V = SparseMat.zeros(n,k)
    buf = np.zeros(n*k_*2)
    d, s, g = np.zeros(n), np.zeros(n*k_), np.zeros(n*k_)

    queue = np.zeros(n, dtype=np.int32)
    is_in = np.zeros(n, dtype=np.int32)

    if comm is None: comm = np.zeros(n, dtype=np.int32)
    else: comm = comm.copy()
    if n_comm is None: n_comm = np.zeros(1, dtype=np.int32)
    else: n_comm = np.array([n_comm], dtype=np.int32)

    fval = _cpp.solve_locale(
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
    Gdiag = np.zeros(n_comm)

    #print(A.indptr.shape, A.indices.shape, A.data.shape, Adiag.shape)
    #print(G.indptr.shape, G.indices.shape, G.data.shape, Gdiag.shape)
    #print(comm.shape)

    _cpp.aggregate_clusters(
            A.indptr, A.indices, A.data, Adiag,
            G.indptr, G.indices, G.data, Gdiag,
            comm)

    return G, Gdiag

def mixing_merge(comm, comm_next):
    #for i, ic in enumerate(comm.tolist()):
    #    comm_next[ic] = comm[i]
    _cpp.merge(comm, comm_next)

def mixing_split(comm, comm_next):
    #for i, ic in enumerate(comm.tolist()):
    #    comm[i] = comm_next[ic]
    _cpp.split(comm, comm_next)

def mixing_locale(A, k=8, eps=1e-6, max_outer=10, max_lv=10, max_inner=2):
    A = SparseMat.from_scipy(A)
    n = len(A.indptr)-1
    Adiag = np.zeros(n)
    comm_init = None
    for it in range(max_outer):
        comms = []
        G, Gdiag = A.copy(), Adiag.copy()
        for lv in range(max_lv):
            print(f'\nouter iter {it+1} lv {lv+1}\n')
            fval, comm, n_comm, V = mixing_cluster(G, Gdiag, k, comm=comm_init, eps=eps, max_iter=max_inner, comm_init=comm_init is not None)
            #print(V)
            print(fval, n_comm)
            if 1:
                fval, new_comm, new_n_comm, V = mixing_cluster(G, Gdiag, k, comm=comm, n_comm=n_comm, eps=1e-4, max_iter=1, shrink=1)
                print(fval, new_n_comm)
            else: # Louvain
                new_comm, new_n_comm = comm.copy(), n_comm

            if new_n_comm == len(comm): break

            comm_init = np.zeros(new_n_comm, dtype=np.int32)
            mixing_merge(comm, comm_init)

            comms.append(new_comm.copy())
            G, Gdiag = mixing_aggregate(G, Gdiag, new_comm, new_n_comm)

        for lv in reversed(range(len(comms)-1)):
            mixing_split(comms[lv], comms[lv+1])
        comm_init = comms[0].copy()

    return comm_init
