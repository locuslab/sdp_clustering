typedef struct SparseMat {
    int *indptr;
    int *indices;
    float *data;
} SparseMat;

typedef struct SparsePair {
    int idx;
    float val;
} SparsePair;

typedef struct Ring {
    int front, rear, len, cap;
    int *queue;
    int *is_in;
} Ring;

float mix_cluster_cpu(int max_iter, float eps, 
        int n, SparseMat A, float *Adiag, SparseMat V,
        SparsePair *buf, float *s, float *d, float *g,
        Ring *Q, int *comm, int *n_comm,
        int shrink, int comm_init);

void mix_aggregate_cpu(int nA, SparseMat A, float *Adiag, int nG, SparseMat G, float *Gdiag, int *comm);
