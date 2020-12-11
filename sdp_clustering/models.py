import numpy as np
from scipy.sparse import csr_matrix
import sdp_clustering._cpp as _cpp

class SparseMat(object):
    def __init__(self, indptr, indices, data):
        self.indptr = indptr
        self.indices = indices
        self.data = data

    def copy(self):
        return SparseMat(self.indptr.copy(), self.indices.copy(), self.data.copy())

    @classmethod
    def from_scipy(cls, A):
        if not A is csr_matrix:
            A = A.tocsr()
        indptr = A.indptr
        indices = A.indices
        data = np.asarray(A.data, dtype=np.float32)

        return cls(indptr, indices, data)

    def to_scipy(self):
        return csr_matrix((self.data, self.indices, self.indptr))

    @classmethod
    def zeros(cls, n, k):
        indptr = np.arange(0, n*k+1, k, dtype=np.int32)
        indices = np.zeros(n*k, dtype=np.int32)
        data = np.zeros(n*k, dtype=np.float32)

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
                if self.data[p].item() == 0.0: continue
                s.append(f'{self.indices[p].item()+1}:{self.data[p].item():1.2f}\t')
            s.append('\n')
        return ''.join(s)

    def savetxt(self, fname):
        f = open(fname, 'w')
        n = self.indptr.shape[0]-1
        for i in range(n):
            for p in range(self.indptr[i], self.indptr[i+1]):
                idx, val = self.indices[p].item()+1, self.data[p].item()
                if val == 0.0: continue
                f.write(f'{idx}:{val}\t')
            f.write('\n')
        f.close()

def init_random_seed(seed):
    _cpp.init_random_seed(seed)

def solve_locale(A, Adiag, k, comm=None, n_comm=None, max_iter=100, eps=1e-3, shrink=0, comm_init=0, rnd_card=1, verbose=False):
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
        shrink, comm_init, rnd_card, verbose)

    return fval, comm, n_comm.item(), V

def aggregate_clusters(A, Adiag, comm, n_comm):
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

def merge_clusters(comm, comm_next, new_comm):
    #for i, ic in enumerate(new_comm.tolist()):
    #    comm_next[ic] = comm[i]
    _cpp.merge(comm, comm_next, new_comm)

def split_clusters(comm, comm_next):
    #for i, ic in enumerate(comm.tolist()):
    #    comm[i] = comm_next[ic]
    _cpp.split(comm, comm_next)

def locale_embedding(A, k=8, eps=1e-6, max_inner=10, verbose=False):
    A = SparseMat.from_scipy(A)
    n = len(A.indptr)-1
    Adiag = np.zeros(n)
    fval, _, _, V = solve_locale(A, Adiag, k, comm=None, eps=eps, max_iter=max_inner, comm_init=False, rnd_card=0, verbose=verbose)
    return V

def leiden_locale(A, k=8, eps=1e-6, max_outer=10, max_lv=10, max_inner=2, verbose=0):
    A = SparseMat.from_scipy(A)
    n = len(A.indptr)-1
    Adiag = np.zeros(n)
    comm_init = None
    for it in range(max_outer):
        comms = []
        G, Gdiag = A.copy(), Adiag.copy()
        for lv in range(max_lv):
            # LocaleEmbedding and LocaleRounding
            fval, comm, n_comm, V = solve_locale(G, Gdiag, k, comm=comm_init, eps=eps, max_iter=max_inner, comm_init=comm_init is not None, verbose=verbose)
            if verbose: print(f'iter {it+1}({lv+1})\topt fval {fval:.8f}\tn_comm {n_comm}')
            if 1: # LeidenRefine
                fval, new_comm, new_n_comm, V = solve_locale(G, Gdiag, k, comm=comm, n_comm=n_comm, eps=1e-4, max_iter=1, shrink=1, verbose=verbose)
                if verbose: print(f'iter {it+1}({lv+1})\trnd fval {fval:.8f}\tn_comm {new_n_comm}')
            else: # If k=1, this branch equals the Louvain algorithm
                new_comm, new_n_comm = comm.copy(), n_comm

            if new_n_comm == len(comm): break
            
            comm_init = np.zeros(new_n_comm, dtype=np.int32)
            merge_clusters(comm, comm_init, new_comm)

            # Aggregrate
            comms.append(new_comm.copy())
            G, Gdiag = aggregate_clusters(G, Gdiag, new_comm, new_n_comm)

        for lv in reversed(range(len(comms)-1)):
            split_clusters(comms[lv], comms[lv+1])
        comm_init = comms[0].copy()
        if verbose: print()

    return comm_init
