#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "cluster.h"

namespace py = pybind11;

using arr = py::array;
float *fptr(arr& a) { return (float*) a.mutable_data(); }
int *iptr(arr& a)   { return (int*) a.mutable_data(); }

static int has_inited = 0;

void randinit()
{
    has_inited = 1;
    long int tt = (long int)time(NULL);
    fprintf(stderr, "tt = %ld\n", tt);
    srand48(tt);
    //srand48(1111);
    srandom(tt);
}

SparseMat SparseMat_init(arr indptr, arr indices, arr data)
{
    SparseMat X;
    X.indptr = iptr(indptr);
    X.indices = iptr(indices);
    X.data = fptr(data);
    return X;
}

float py_solve_locale(int max_iter, float eps,
        arr Aindptr, arr Aindices, arr Adata, arr Adiag,
        arr Vindptr, arr Vindices, arr Vdata,
        arr buf, arr s, arr d, arr g,
        arr queue, arr is_in,
        arr comm, arr n_comm,
        int shrink, int comm_init)
{
    if (!has_inited) randinit();
    SparseMat A = SparseMat_init(Aindptr, Aindices, Adata);
    SparseMat V = SparseMat_init(Vindptr, Vindices, Vdata);

    int n = Aindptr.shape(0)-1;
    Ring Q = {0, 0, 0, n, iptr(queue), iptr(is_in)};

    float fval = solve_locale(max_iter, eps, 
        n, A, fptr(Adiag), V,
        (SparsePair*)buf.mutable_data(), fptr(s), fptr(d), fptr(g),
        &Q,
        iptr(comm), iptr(n_comm),
        shrink, comm_init);
    return fval;
}

void py_aggregate_clusters(
        arr Aindptr, arr Aindices, arr Adata, arr Adiag,
        arr Gindptr, arr Gindices, arr Gdata, arr Gdiag,
        arr comm)
{
    SparseMat A = SparseMat_init(Aindptr, Aindices, Adata);
    SparseMat G = SparseMat_init(Gindptr, Gindices, Gdata);

    int nA = Aindptr.shape(0)-1;
    int nG = Gindptr.shape(0)-1;
    aggregate_clusters(
            nA, A, fptr(Adiag),
            nG, G, fptr(Gdiag),
            iptr(comm));
}

void py_merge(arr comm, arr comm_next)
{
    int n = comm.shape(0);
    merge(n, iptr(comm), iptr(comm_next));
}

void py_split(arr comm, arr comm_next)
{
    int n = comm.shape(0);
    split(n, iptr(comm), iptr(comm_next));
}

PYBIND11_MODULE(EXTENSION_NAME, m) {
    m.def("solve_locale" ,  &py_solve_locale,  "Solve locale optimization");
    m.def("aggregate_clusters" , &py_aggregate_clusters, "Form hypergraph");
    m.def("merge" ,  &py_merge,  "Merge (cpu)");
    m.def("split" ,  &py_split,  "Split (cpu)");
}
