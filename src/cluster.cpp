#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <algorithm>

#include <stdint.h>
#ifndef __unix__
#include <sys/time.h>
#endif

#include "cluster.h"

#define MEPS 1e-20

inline int min(int x, int y) { return (x<=y)?x:y; }

#define NS_PER_SEC 1000000000
int64_t wall_clock_ns()
{
#ifdef __unix__
    struct timespec tspec;
    int r = clock_gettime(CLOCK_MONOTONIC, &tspec);
    assert(r==0);
    return tspec.tv_sec*NS_PER_SEC + tspec.tv_nsec;
#else
    struct timeval tv;
    int r = gettimeofday( &tv, NULL );
    assert(r==0);
    return tv.tv_sec*NS_PER_SEC + tv.tv_usec*1000;
#endif
}

double wall_time_diff(int64_t ed, int64_t st)
{
    return (double)(ed-st)/(double)NS_PER_SEC;
}

// ------------- Sparse BLAS-like utils --------------------

// perform y += a * xi for sparse matrix X
void axipy(float *__restrict__ y, float a, SparseMat X, int i)
{
    int st = X.indptr[i], ed = X.indptr[i+1];
    const int *__restrict__ indices = X.indices;
    const float *__restrict__ data = X.data;

    for (int p=st; p<ed; p++) {
        y[indices[p]] += a * data[p];
    }
}

// perform y[idx] += a * x[idx] for all idx in sparse si,
// and gather the item to SparsePair *buf if it is positive
// NOTE: assume a*x to be non-positive
void axpy_and_gather(float *__restrict__ y, float a, float *__restrict__ x, SparseMat S, int i, SparsePair *buf, int *nbuf)
{
    int st = S.indptr[i], ed = S.indptr[i+1];
    int *__restrict__ indices = S.indices;

    for (int p=st; p<ed; p++) {
        int idx = indices[p];
        float r = y[idx];
        if (r==0) continue;

        r += a * x[idx];
        buf[(*nbuf)++] = {idx, r};

        y[idx] = 0;
    }
}

float ydotxi(float *__restrict__ y, SparseMat X, int i)
{
    int st = X.indptr[i], ed = X.indptr[i+1];
    int *__restrict__ indices = X.indices;
    float *__restrict__ data = X.data;

    float s=0;
    for (int p=st; p<ed; p++)
        s += y[indices[p]] * data[p];

    return s;
}

// Assume indices are ordered
float xidotxj(SparseMat X, int i, int j)
{
    int ist = X.indptr[i], ied = X.indptr[i+1];
    int jst = X.indptr[j], jed = X.indptr[j+1];
    int *__restrict__ indices = X.indices;
    float *__restrict__ data = X.data;

    float s=0;
    for (int p=ist; p<ied; p++) {
        for (int q=jst; q<jed; q++) {
            if (indices[p] == indices[q])
                s += data[p] * data[q];
        }
    }

    return s;
}

float snrm2_pairs(SparsePair *buf, int k)
{
    float s = 0;
    for (int i=0; i<k; i++)
        s += buf[i].val * buf[i].val;

    return sqrt(s);
}

#if 0
int val_cmp(const void *px, const void *py) 
{
    float vx = ((SparsePair*)px)->val, vy = ((SparsePair*)py)->val;

    if (vx < vy) return 1;
    else if (vx > vy) return -1;
    else return 0;
}
#else
bool val_cmp(const SparsePair &x, const SparsePair &y) 
{
    return x.val > y.val;
}
#endif

int idx_cmp(const void *px, const void *py) 
{
    int vx = ((SparsePair*)px)->idx, vy = ((SparsePair*)py)->idx;

    if (vx < vy) return -1;
    else if (vx > vy) return 1;
    else return 0;
}

// ------------- randomization utils -------------------

void randperm(int *perm, int k)
{
    for (int i=0; i<k; i++)
        perm[i] = i;

    for(int i=0; i<k; i++){
        int j = (int)random() % (k-i);
        int t = perm[i];
        perm[i] = perm[j];
        perm[j] = t;
    }
}

void randpick(SparsePair *buf, int n, float *g, float off)
{
    float s = 0;
    float theta = 1;
    int i;
    float mx = buf[0].val;
    for (i=0; i<n; i++) {
        if (buf[i].val-off < 0) break;
        //fprintf(stderr, "%d %f\n", buf[i].idx, diff);
        float gi = expf((buf[i].val - mx) / theta);
        s += gi;
        g[i] = gi;
    }
    if (i==0) return;
    //n = i+1;
    float r = drand48() * s;
    for (i=0; i<n; i++) {
        r -= g[i];
        if (r<=0 || n==1) break;
    }
    if (i==n) {
        fprintf(stderr, "bug! r %f s %f n %d\n", r, s, n);
        i = n-1;
        //exit(0);
    }
    buf[0] = buf[i];

    for (i=0; i<n; i++) g[i] = 0;
}

// -------- Ring methods ----------------

void ring_push(Ring *Q, int x)
{
    if (Q->is_in[x]) return;
    Q->queue[Q->rear] = x;
    Q->rear = (Q->rear + 1) % Q->cap; 
    Q->len++;
    Q->is_in[x] = 1;
}

int ring_pop(Ring *Q)
{
    int x = Q->queue[Q->front];
    Q->front = (Q->front + 1) % Q->cap;
    Q->len--;
    Q->is_in[x] = 0;

    return x;
}

void ring_reset(Ring *Q)
{
    //for (int i=0; i<Q->cap; i++) Q->queue[i] = i;
    randperm(Q->queue, Q->cap);
    Q->len = Q->cap;
    Q->front = Q->rear = 0;
    for (int i=0; i<Q->cap; i++) Q->is_in[i] = 1;
}

/* -------------- main algorihtm ------------------------*/

float solve_locale(int max_iter, float eps, 
        int n, SparseMat A, float *Adiag, SparseMat V,
        SparsePair *buf, float *__restrict__ s, float *__restrict__ d, float *__restrict__ g,
        Ring *Q, int *comm, int *n_comm, int shrink, int comm_init)
{
    int64_t time_st = wall_clock_ns();
    fprintf(stderr, "n_comm %d\n", *n_comm);
    double fval = 0;
    double m = 0;
    for (int i=0; i<n; i++) {
        // init vi = ei
        V.indices[V.indptr[i]] = comm_init?comm[i]:i;
        V.data[V.indptr[i]] = 1;

        // di = sum_j aij
        d[i] = Adiag[i];
        for (int p=A.indptr[i]; p<A.indptr[i+1]; p++)
            d[i] += A.data[p];
        m += d[i];

        // s = sum_i di vi
        s[V.indices[V.indptr[i]]] += d[i];
    }
    for (int i=0; i<n; i++)
        fval -= s[i]*s[i];

    m /= 2; // summing symmetric edge only once
    fval /= 2*m;
    for (int i=0; i<n; i++) fval += Adiag[i];

    if (comm_init) {
        for (int i=0; i<n; i++) {
            for (int p=A.indptr[i]; p<A.indptr[i+1]; p++) {
                if (comm[i] == comm[A.indices[p]]) fval += A.data[p];
            }
        }
        memset(comm, 0, n * sizeof *comm);
    } 

    fval /= 2*m;
    int k = V.indptr[1]-V.indptr[0];
    fprintf(stderr, "k = %d m %lf\n", k, m);

    int64_t time_now = wall_clock_ns();
    fprintf(stderr, "iter 0 fval %f time %.4e\n", fval, wall_time_diff(time_now, time_st));

    int iter=0, first=0, nvisited=0;
    double delta = 0;
    int rank = n;

    ring_reset(Q);
    while (1) {
        if (nvisited >= n || Q->len == 0) {
            //if (iter>2*max_iter) break; // XXX
            iter ++;
            fval += delta/(2*m);
            time_now = wall_clock_ns();
            fprintf(stderr, "iter %d fval %.8e delta %.2e %s %s qlen %d time %.4e\n", iter, fval, delta/(2*m), shrink?"shrink":"", first?"first":"", Q->len, wall_time_diff(time_now, time_st));
            double scaled_delta = fabs(delta/(2*m));
            if ((!shrink && (scaled_delta < eps || iter>=max_iter)) || (shrink && (scaled_delta < eps*1e-2 || Q->len==0 || iter>=15))) { // practice
            //if ((!shrink && (scaled_delta < eps || iter>=max_iter)) || (shrink && (Q->len==0 || iter>=30))) { // validation
                //if (shrink && (*n_comm || iter>=max_iter)) break;
                if (shrink) break;
                shrink ^= 1;
                first = shrink;
                ring_reset(Q);
            } else {
                first = 0;
            }
            //if (Q->len==0) ring_reset(Q);

            delta = 0, nvisited = 0;
            //ring_reset(Q); // NOTE Enable when validation
        }

        int i = ring_pop(Q);
        int ic = comm[i];
        nvisited++;

        if (A.indptr[i]==A.indptr[i+1]) continue;
        if (*n_comm) {
            int nonsingleton = 0;
            for (int p=A.indptr[i]; p<A.indptr[i+1]; p++) {
                int j = A.indices[p];
                if (ic == comm[j] && V.indices[V.indptr[i]] == V.indices[V.indptr[j]]) {
                    nonsingleton = 1;
                    break;
                }
            }
            if (nonsingleton) continue;
        }

        // g = sum_j aij vj
        for (int p=A.indptr[i]; p<A.indptr[i+1]; p++) {
            int j = A.indices[p];
            if (ic == comm[j]) axipy(g, A.data[p], V, j);
        }
        axipy(s, -d[i], V, i);

        // old_gv = g'vi
        float old_gv = ydotxi(g, V, i) - d[i]/(2*m) * ydotxi(s, V, i);

        // gather positive parts of g into buf
        int nbuf = 0;
        for (int p=A.indptr[i]; p<A.indptr[i+1]; p++) {
            int j = A.indices[p];
            if (ic == comm[j]) axpy_and_gather(g, -d[i]/(2*m), s, V, j, buf, &nbuf);
        }

        // get top coordinates from buf to vi
        //qsort(buf, nbuf, sizeof *buf, val_cmp);
        int mid = (nbuf>k)? k : nbuf;
        std::partial_sort(buf, buf+mid, buf+nbuf, val_cmp);
        int npos = nbuf;
        for (int q=nbuf-1; q>=0; q--)
            if (buf[q].val <= 0) npos--;

        float gv, gnrm;
        if (npos == 0) { // gv=0 at best
            if (old_gv > -MEPS) {           // vi'g = 0 -> g[indices of vi] = 0
                buf[0] = {V.indices[V.indptr[i]], V.data[V.indptr[i]]};
            } else if (buf[0].val == 0) {   // g0 = 0
                buf[0].val = 1;
            } else {                        // increase rank
                buf[0] = {rank++, 1};
                if(rank > n*10) {fprintf(stderr, "rank explode\n"); exit(0);}
            }
            gnrm = 1, gv = 0, npos = 1;
        } else { // update
            //if (shrink) randpick(buf, npos, g, first?-100:old_gv);
            npos = min(npos, V.indptr[i+1]-V.indptr[i]);
            if (shrink) npos = 1;
            gv = gnrm = snrm2_pairs(buf, npos);
        }
        
        if ( (gv - old_gv <= MEPS) && !first ) { // if not increasing, continue
            axipy(s, d[i], V, i);
            continue;
        }

        if (gnrm <= MEPS) {
            fprintf(stderr, "nbuf %d npos %d v0 %e v1 %e buf0 %g %d\n", nbuf, npos, V.data[V.indptr[i]], V.data[V.indptr[i]+1], buf[0].val, buf[0].val<0);
            fprintf(stderr, "i=%d gnrm %e gv %e old_gv %e gv-old_gv %e first %d shrink %d\n", i, gnrm, gv, old_gv, gv-old_gv, first, shrink);
            exit(0);
        }
        // copy vector from buf
        for (int p=V.indptr[i], q=0; p<V.indptr[i+1]; p++, q++) {
            if (q >= npos) buf[q] = {0, 0};
            V.indices[p] = buf[q].idx;
            V.data[p] = buf[q].val / gnrm;
        }
        axipy(s, d[i], V, i);

        delta += (gv - old_gv)*2;

        for (int p=A.indptr[i]; p<A.indptr[i+1]; p++) {
            int j = A.indices[p];
            if (ic == comm[j]) ring_push(Q, j);
        }
    }

    // renumber communities
    *n_comm = 0;
    memset(g, 0, rank * sizeof *g);
    for (int i=0; i<n; i++) {
        int raw_ic = V.indices[V.indptr[i]];
        if (g[raw_ic] == 0) {
            buf[raw_ic].idx = (*n_comm)++;
            g[raw_ic] = 1;
        }
        comm[i] = buf[raw_ic].idx;
    }

    return fval;
}

void aggregate_clusters(int nA, SparseMat A, float *Adiag, int nG, SparseMat G, float *Gdiag, int *comm)
{
    for (int i=0; i<nA; i++) {
        int ic = comm[i];
        for (int p=A.indptr[i]; p<A.indptr[i+1]; p++) {
            int jc = comm[A.indices[p]];
            if (ic == jc) continue;
            G.indptr[ic+1]++;
        }
    }
    for (int i=0; i<nG; i++) {
        G.indptr[i+1] += G.indptr[i];
    }
    for (int i=0; i<nA; i++) {
        int ic = comm[i];
        Gdiag[ic] += Adiag[i];
        for (int p=A.indptr[i]; p<A.indptr[i+1]; p++) {
            int jc = comm[A.indices[p]]; 
            if (ic == jc) {
                Gdiag[ic] += A.data[p];
                continue;
            }
            int pic = G.indptr[ic];
            G.indices[pic] = jc, G.data[pic] = A.data[p];
            G.indptr[ic]++;
        }
    }
    for (int i=nG-1; i>=0; i--) {
        G.indptr[i+1] = G.indptr[i];
    }
    G.indptr[0] = 0;

#if 0
    double mA = 0;
    for (int i=0; i<nA; i++) {
        mA += Adiag[i];
        for (int p=A.indptr[i]; p<A.indptr[i+1]; p++) {
            mA += A.data[p];
        }
    }

    double mG = 0;
    for (int i=0; i<nG; i++) {
        mG += Gdiag[i];
        for (int p=G.indptr[i]; p<G.indptr[i+1]; p++) {
            mG += G.data[p];
        }
    }

    fprintf(stderr, "mA %lf mG %lf\n", mA, mG);
#endif
}

void merge(int n, int *comm, int *comm_next) 
{
    for (int i=0; i<n; i++)
        comm_next[comm[i]] = comm[i];
}

void split(int n, int *comm, int *comm_next)
{
    for (int i=0; i<n; i++)
        comm[i] = comm_next[comm[i]];
}
