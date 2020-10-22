#include <torch/extension.h>
#include "cluster.h"

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

SparseMat SparseMat_init(at::Tensor indptr, at::Tensor indices, at::Tensor data)
{
    SparseMat X;
    X.indptr = indptr.data_ptr<int>();
    X.indices = indices.data_ptr<int>();
    X.data = data.data_ptr<float>();
    return X;
}

float mix_cluster(int max_iter, float eps,
        at::Tensor Aindptr, at::Tensor Aindices, at::Tensor Adata, at::Tensor Adiag,
        at::Tensor Vindptr, at::Tensor Vindices, at::Tensor Vdata,
        at::Tensor buf, at::Tensor s, at::Tensor d, at::Tensor g,
        at::Tensor queue, at::Tensor is_in,
        at::Tensor comm, at::Tensor n_comm,
        int shrink, int comm_init)
{
    if (!has_inited) randinit();
    SparseMat A = SparseMat_init(Aindptr, Aindices, Adata);
    SparseMat V = SparseMat_init(Vindptr, Vindices, Vdata);

    int n = Aindptr.size(0)-1;
    Ring Q = {0, 0, 0, n, queue.data_ptr<int>(), is_in.data_ptr<int>()};

    float fval = mix_cluster_cpu(max_iter, eps, 
        n, A, Adiag.data_ptr<float>(), V,
        (SparsePair*)buf.data_ptr<float>(), s.data_ptr<float>(), d.data_ptr<float>(), g.data_ptr<float>(),
        &Q,
        comm.data_ptr<int>(), n_comm.data_ptr<int>(),
        shrink, comm_init);
    return fval;
}

void mix_aggregate(
        at::Tensor Aindptr, at::Tensor Aindices, at::Tensor Adata, at::Tensor Adiag,
        at::Tensor Gindptr, at::Tensor Gindices, at::Tensor Gdata, at::Tensor Gdiag,
        at::Tensor comm)
{
    SparseMat A = SparseMat_init(Aindptr, Aindices, Adata);
    SparseMat G = SparseMat_init(Gindptr, Gindices, Gdata);

    int nA = Aindptr.size(0)-1;
    int nG = Gindptr.size(0)-1;
    mix_aggregate_cpu(
            nA, A, Adiag.data_ptr<float>(),
            nG, G, Gdiag.data_ptr<float>(),
            comm.data_ptr<int>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cluster" ,  &mix_cluster,  "Mix cluster (cpu)");
    m.def("aggregate" , &mix_aggregate, "Mix aggregate (cpu)");
}
