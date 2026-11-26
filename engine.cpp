// YOU MAY EDIT THIS FILE

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "engine.h"

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();
constexpr int LANES = 4;
constexpr int STRIDE = 8;
constexpr int CHUNK_MAX = 128;
constexpr size_t BLOCK_BYTES = 32 << 10;

enum { TAG_REQ = 11, TAG_WORK = 12, TAG_RES = 13 };

struct Cand {
    double d;
    int i;
};

// equal distances favour the larger point index
inline bool worse(const Cand &a, const Cand &b) {
    return a.d > b.d || (a.d == b.d && a.i < b.i);
}

inline bool better(const Cand &a, const Cand &b) {
    return worse(b, a);
}

// max-heap of the k best; root is the worst retained neighbour so new points reject with one compare
struct TopK {
    Cand *h = nullptr;
    int k = 0, n = 0;
    double kth = INF;

    void reset(Cand *storage, int kk) {
        h = storage;
        k = kk;
        n = 0;
        kth = INF;
    }

    void offer(double d, int i) {
        Cand c{d, i};

        if (n == k) {
            if (!worse(h[0], c)) return;

            int idx = 0;
            for (;;) {
                int l = 2 * idx + 1, r = l + 1, m = idx;
                const Cand *best = &c;

                if (l < n && worse(h[l], *best)) {
                    m = l;
                    best = &h[l];
                }
                if (r < n && worse(h[r], *best)) {
                    m = r;
                    best = &h[r];
                }
                if (m == idx) break;

                h[idx] = h[m];
                idx = m;
            }

            h[idx] = c;
            kth = h[0].d;
        } else {
            int idx = n++;

            while (idx > 0 && worse(c, h[(idx - 1) / 2])) {
                h[idx] = h[(idx - 1) / 2];
                idx = (idx - 1) / 2;
            }

            h[idx] = c;
            if (n == k) kth = h[0].d;
        }
    }

    void finish() {
        std::sort(h, h + n, better);
    }
};

// equal vote counts favour the larger label
int vote(const Cand *nb, int k, const int *labels, std::vector<int> &tmp) {
    tmp.resize(k);
    for (int i = 0; i < k; ++i) tmp[i] = labels[nb[i].i];
    std::sort(tmp.begin(), tmp.end());

    int win = 0, most = 0;

    for (int i = 0; i < k;) {
        int lab = tmp[i], votes = 0;

        while (i < k && tmp[i] == lab) {
            ++votes;
            ++i;
        }

        if (votes > most || (votes == most && lab > win)) {
            win = lab;
            most = votes;
        }
    }

    return win;
}

// enables streaming and early-out of 4-point groups
struct Scan {
    const double *blk;
    const double *q;
    int B, npts, base, d;
};

inline void offer_group(const double *s, int lim, int first, TopK &H) {
    for (int j = 0; j < lim; ++j)
        if (s[j] <= H.kth) H.offer(s[j], first + j);
}

// four independent accumulators so we can early-out a whole group of 4
// points without destroying the per-point order needed for tie-breaks
void scan(const Scan &A, TopK &H) {
    const int B = A.B, d = A.d;
    const double *q = A.q;

    for (int p = 0; p < A.npts; p += 4) {
        const int lim = std::min(4, A.npts - p);
        const double *x = A.blk + p;
        const double kth = H.kth;

        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int a = 0;
        bool dead = false;

#define STEP { double qa = q[a]; double t0 = qa - x[0], t1 = qa - x[1], t2 = qa - x[2], t3 = qa - x[3]; \
               s0 += t0 * t0; s1 += t1 * t1; s2 += t2 * t2; s3 += t3 * t3; x += B; ++a; }

        while (a + STRIDE <= d) {
            STEP STEP STEP STEP STEP STEP STEP STEP

            // strict > so equal partial distances still reach the final comparison for tie-breaker
            if (s0 > kth && s1 > kth && s2 > kth && s3 > kth) {
                dead = true;
                break;
            }
        }

        if (dead) continue;
        while (a < d) STEP

#undef STEP

        const double s[4] = {s0, s1, s2, s3};
        offer_group(s, lim, A.base + p, H);
    }
}

struct Hdr {
    int32_t mode, bits, e, pad;
    int64_t minv, count;
};

const double POW10[10] = {
    1e0, 1e1, 1e2, 1e3, 1e4,
    1e5, 1e6, 1e7, 1e8, 1e9
};

inline double decode(int64_t m, int e) {
    return (double)m / POW10[e];
}

inline size_t pack_capacity(size_t n) {
    return sizeof(Hdr) + n * sizeof(double);
}

// use raw doubles when decimal packing cannot reproduce every value
size_t pack(const double *v, size_t n, char *out, bool allow) {
    Hdr h = {0, 0, 0, 0, 0, (int64_t)n};
    std::vector<int64_t> m(allow ? n : 0);

    for (int e = 0; allow && n > 0 && e < 10 && h.mode == 0; ++e) {
        int64_t lo = 0, hi = 0;
        size_t i = 0;

        for (; i < n; ++i) {
            double y = v[i] * POW10[e];

            if (!(std::fabs(y) < 4e15)) break;

            m[i] = (int64_t)std::llrint(y);
            if (decode(m[i], e) != v[i]) break;

            lo = i ? std::min(lo, m[i]) : m[i];
            hi = i ? std::max(hi, m[i]) : m[i];
        }

        if (i < n) continue;

        int bits = 0;
        while (bits < 64 && ((uint64_t)(hi - lo) >> bits) != 0)
            ++bits;

        if (bits <= 56) {
            h.mode = 1;
            h.bits = bits;
            h.e = e;
            h.minv = lo;
        }
    }

    std::memcpy(out, &h, sizeof h);
    char *w = out + sizeof h;

    if (h.mode == 0) {
        std::memcpy(w, v, n * sizeof(double));
        return pack_capacity(n);
    }

    if (h.bits == 0) return sizeof h;

    const size_t words = (n * h.bits + 63) / 64;
    std::vector<uint64_t> buf(words, 0);
    const uint64_t mask = (1ULL << h.bits) - 1;

    for (size_t i = 0, bit = 0; i < n; ++i, bit += h.bits) {
        uint64_t u = (uint64_t)(m[i] - h.minv) & mask;

        buf[bit / 64] |= u << (bit % 64);

        if (bit % 64 + h.bits > 64)
            buf[bit / 64 + 1] |= u >> (64 - bit % 64);
    }

    std::memcpy(w, buf.data(), words * 8);
    return sizeof h + words * 8;
}

void unpack(const char *in, double *v, size_t n) {
    Hdr h;
    std::memcpy(&h, in, sizeof h);

    if ((size_t)h.count != n) MPI_Abort(MPI_COMM_WORLD, 4);

    const char *r = in + sizeof h;

    if (h.mode == 0) {
        std::memcpy(v, r, n * sizeof(double));
        return;
    }

    if (h.bits == 0) {
        for (size_t i = 0; i < n; ++i)
            v[i] = decode(h.minv, h.e);
        return;
    }

    const size_t words = (n * h.bits + 63) / 64;
    std::vector<uint64_t> buf(words);
    std::memcpy(buf.data(), r, words * 8);

    const uint64_t mask = (1ULL << h.bits) - 1;

    for (size_t i = 0, bit = 0; i < n; ++i, bit += h.bits) {
        uint64_t u = buf[bit / 64] >> (bit % 64);

        if (bit % 64 + h.bits > 64)
            u |= buf[bit / 64 + 1] << (64 - bit % 64);

        v[i] = decode(h.minv + (int64_t)(u & mask), h.e);
    }
}

int env_int(const char *name, int def) {
    const char *v = std::getenv(name);
    return (v && *v) ? std::atoi(v) : def;
}

}  // namespace

void Engine::KNN(Params &p,
                 std::vector<DataPoint> &dataset,
                 std::vector<Query> &queries) {
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int hdr[5] = {0};

    if (rank == 0) {
        int kmax = 0;
        for (const Query &q : queries)
            kmax = std::max(kmax, q.k);

        int v[5] = {
            p.num_data,
            p.num_queries,
            p.num_attrs,
            kmax,
            env_int("KNN_COMPRESS", 1)
        };

        std::memcpy(hdr, v, sizeof v);
    }

    MPI_Bcast(hdr, 5, MPI_INT, 0, MPI_COMM_WORLD);

    const int N = hdr[0], Q = hdr[1], d = hdr[2], kmax = hdr[3];
    const bool compress = hdr[4] != 0;

    if (N <= 0 || Q <= 0 || d <= 0) return;

    int B = (int)(BLOCK_BYTES / ((size_t)d * sizeof(double)));
    B = std::max(LANES, B - B % LANES);

    const int nblocks = (N + B - 1) / B;
    const int chunk_cap = std::max(
        1,
        std::min(
            CHUNK_MAX,
            (int)((64u << 20) / (sizeof(Cand) * (size_t)kmax))
        )
    );

    std::vector<int> label_storage(N);

    // zero padding lets the kernel read four lanes in the final block
    std::vector<double> dataset_storage((size_t)nblocks * B * d, 0.0);

    if (rank == 0) {
        for (int i = 0; i < N; ++i)
            label_storage[i] = dataset[i].label;

        for (int b = 0; b < nblocks; ++b) {
            double *blk = dataset_storage.data() + (size_t)b * B * d;
            const int np = std::min(B, N - b * B);

            for (int l = 0; l < np; ++l) {
                const double *src = dataset[b * B + l].attrs.data();

                for (int a = 0; a < d; ++a)
                    blk[(size_t)a * B + l] = src[a];
            }
        }
    }

    MPI_Bcast(label_storage.data(), N, MPI_INT, 0, MPI_COMM_WORLD);

    // split large buffers so each mpi count fits in an int
    const size_t count_limit = (size_t)std::numeric_limits<int>::max();

    for (size_t offset = 0; offset < dataset_storage.size();) {
        const int count = (int)std::min(
            dataset_storage.size() - offset,
            count_limit
        );

        MPI_Bcast(
            dataset_storage.data() + offset,
            count,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        offset += (size_t)count;
    }

    const int *labels = label_storage.data();
    const double *X = dataset_storage.data();

    const size_t work_max =
        ((2 + chunk_cap) * sizeof(int) + 7) / 8 * 8 +
        pack_capacity((size_t)chunk_cap * d);

    std::vector<int> chunk_k(chunk_cap), chunk_labels, tmp;
    std::vector<double> chunk_q((size_t)chunk_cap * d);
    std::vector<double> qscratch((size_t)chunk_cap * d);
    std::vector<TopK> heaps(chunk_cap);
    std::vector<Cand> arena;

    auto pack_work = [&](char *w, int start, int count) {
        std::memcpy(w, &start, 4);
        std::memcpy(w + 4, &count, 4);

        for (int i = 0; i < count; ++i)
            std::memcpy(w + 8 + 4 * i, &queries[start + i].k, 4);

        const size_t hb = ((2 + count) * 4 + 7) / 8 * 8;
        if (count == 0) return hb;

        for (int i = 0; i < count; ++i) {
            std::memcpy(
                &qscratch[(size_t)i * d],
                queries[start + i].attrs.data(),
                d * sizeof(double)
            );
        }

        return hb + pack(
            qscratch.data(),
            (size_t)count * d,
            w + hb,
            compress
        );
    };

    auto unpack_work = [&](const char *w, int &start, int &count) {
        std::memcpy(&start, w, 4);
        std::memcpy(&count, w + 4, 4);

        if (count == 0) return;

        std::memcpy(chunk_k.data(), w + 8, 4 * count);

        unpack(
            w + ((2 + count) * 4 + 7) / 8 * 8,
            chunk_q.data(),
            (size_t)count * d
        );
    };

    auto process_chunk = [&](int count, auto &&between_blocks) {
        size_t need = 0;
        for (int i = 0; i < count; ++i)
            need += chunk_k[i];

        if (arena.size() < need) arena.resize(need);

        for (int i = 0, off = 0; i < count; off += chunk_k[i++])
            heaps[i].reset(arena.data() + off, chunk_k[i]);

        for (int b = 0; b < nblocks; ++b) {
            Scan A;
            A.blk = X + (size_t)b * B * d;
            A.B = B;
            A.npts = std::min(B, N - b * B);
            A.base = b * B;
            A.d = d;

            for (int i = 0; i < count; ++i) {
                A.q = &chunk_q[(size_t)i * d];
                scan(A, heaps[i]);
            }

            // keep work requests and prefetched batches moving during computation
            between_blocks();
        }

        chunk_labels.resize(count);

        for (int i = 0; i < count; ++i) {
            heaps[i].finish();
            chunk_labels[i] = vote(
                heaps[i].h,
                heaps[i].n,
                labels,
                tmp
            );
        }
    };

    auto pack_results = [&](int start, int count, std::vector<char> &buf) {
        size_t bytes = 8;

        for (int i = 0; i < count; ++i)
            bytes += 4 + 4 * (size_t)heaps[i].n;

#ifdef DEBUG
        for (int i = 0; i < count; ++i)
            bytes += 8 * (size_t)heaps[i].n;
#endif

        buf.resize(bytes);
        char *w = buf.data();

        std::memcpy(w, &start, 4);
        std::memcpy(w + 4, &count, 4);
        w += 8;

        for (int i = 0; i < count; ++i) {
            std::memcpy(w, &chunk_labels[i], 4);
            w += 4;

            for (int j = 0; j < heaps[i].n; ++j) {
                std::memcpy(w, &heaps[i].h[j].i, 4);
                w += 4;
            }

#ifdef DEBUG
            for (int j = 0; j < heaps[i].n; ++j) {
                std::memcpy(w, &heaps[i].h[j].d, 8);
                w += 8;
            }
#endif
        }
    };

    std::vector<int> res_label(rank == 0 ? Q : 0);
    std::vector<std::vector<std::pair<double, int>>> res_nb(
        rank == 0 ? Q : 0
    );

    int queries_done = 0, next_query = 0, workers_finished = 0;

    auto store_results = [&](const char *r) {
        int start, count;

        std::memcpy(&start, r, 4);
        std::memcpy(&count, r + 4, 4);
        r += 8;

        for (int i = 0; i < count; ++i) {
            const int qid = start + i, k = queries[qid].k;

            std::memcpy(&res_label[qid], r, 4);
            r += 4;

            res_nb[qid].assign(k, {0.0, 0});

            for (int j = 0; j < k; ++j) {
                std::memcpy(&res_nb[qid][j].second, r, 4);
                r += 4;
            }

#ifdef DEBUG
            for (int j = 0; j < k; ++j) {
                std::memcpy(&res_nb[qid][j].first, r, 8);
                r += 8;
            }
#endif
        }

        queries_done += count;
    };

    // shrink final batches so ranks finish together instead of one holding a large last chunk
    auto take_chunk = [&](int &start, int &count) {
        start = next_query;

        count = std::min(
            Q - start,
            std::max(1, std::min(chunk_cap, (Q - start) / (8 * P)))
        );

        next_query += count;
    };

    struct Pending {
        std::vector<char> buf;
        MPI_Request req;
    };

    std::vector<Pending> pending;
    std::vector<char> wbuf(rank == 0 ? (size_t)P * work_max : 0);
    std::vector<MPI_Request> wreq(
        rank == 0 ? P : 0,
        MPI_REQUEST_NULL
    );

    auto service = [&]() {
        for (;;) {
            int flag;
            MPI_Status st;

            MPI_Iprobe(
                MPI_ANY_SOURCE,
                MPI_ANY_TAG,
                MPI_COMM_WORLD,
                &flag,
                &st
            );

            if (!flag) break;

            const int src = st.MPI_SOURCE;

            if (st.MPI_TAG == TAG_REQ) {
                MPI_Recv(
                    nullptr, 0, MPI_BYTE, src, TAG_REQ,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE
                );

                // previous Isend to this worker must complete before we overwrite its slot in wbuf
                MPI_Wait(&wreq[src], MPI_STATUS_IGNORE);

                int s, c;
                take_chunk(s, c);

                if (c == 0) ++workers_finished;

                char *w = &wbuf[(size_t)src * work_max];

                MPI_Isend(
                    w,
                    (int)pack_work(w, s, c),
                    MPI_BYTE,
                    src,
                    TAG_WORK,
                    MPI_COMM_WORLD,
                    &wreq[src]
                );
            } else {
                int bytes;
                MPI_Get_count(&st, MPI_BYTE, &bytes);

                pending.push_back({
                    std::vector<char>(bytes),
                    MPI_REQUEST_NULL
                });

                MPI_Irecv(
                    pending.back().buf.data(),
                    bytes,
                    MPI_BYTE,
                    src,
                    TAG_RES,
                    MPI_COMM_WORLD,
                    &pending.back().req
                );
            }
        }

        for (size_t i = 0; i < pending.size();) {
            int done;
            MPI_Test(&pending[i].req, &done, MPI_STATUS_IGNORE);

            if (!done) {
                ++i;
                continue;
            }

            store_results(pending[i].buf.data());

            if (i + 1 != pending.size())
                pending[i] = std::move(pending.back());

            pending.pop_back();
        }
    };

    if (rank == 0) {
        auto hook = [&]() {
            service();
        };

        while (queries_done < Q || workers_finished < P - 1) {
            hook();

            int s, c;
            take_chunk(s, c);

            if (c == 0) continue;

            for (int i = 0; i < c; ++i) {
                chunk_k[i] = queries[s + i].k;

                std::memcpy(
                    &chunk_q[(size_t)i * d],
                    queries[s + i].attrs.data(),
                    d * sizeof(double)
                );
            }

            process_chunk(c, hook);

            std::vector<char> buf;
            pack_results(s, c, buf);
            store_results(buf.data());
        }

        for (int qid = 0; qid < Q; ++qid)
            reportResult(queries[qid], res_nb[qid], res_label[qid]);
    } else {
        std::vector<char> wb[2] = {
            std::vector<char>(work_max),
            std::vector<char>(work_max)
        };

        std::vector<char> rb[2];

        MPI_Request rreq[2] = {
            MPI_REQUEST_NULL,
            MPI_REQUEST_NULL
        };
        MPI_Request req = MPI_REQUEST_NULL;

        int wi = 0, ri = 0;

        auto hook = [&]() {
            int f;
            MPI_Test(&req, &f, MPI_STATUS_IGNORE);
        };

        MPI_Send(
            nullptr, 0, MPI_BYTE, 0, TAG_REQ, MPI_COMM_WORLD
        );

        MPI_Irecv(
            wb[wi].data(),
            (int)work_max,
            MPI_BYTE,
            0,
            TAG_WORK,
            MPI_COMM_WORLD,
            &req
        );

        for (;;) {
            MPI_Wait(&req, MPI_STATUS_IGNORE);

            int s, c;
            unpack_work(wb[wi].data(), s, c);

            if (c == 0) break;

            // post the next receive now so it overlaps with the scan of the batch just unpacked
            wi ^= 1;

            MPI_Send(
                nullptr, 0, MPI_BYTE, 0, TAG_REQ, MPI_COMM_WORLD
            );

            MPI_Irecv(
                wb[wi].data(),
                (int)work_max,
                MPI_BYTE,
                0,
                TAG_WORK,
                MPI_COMM_WORLD,
                &req
            );

            process_chunk(c, hook);

            // finish the older send before reusing its result buffer
            MPI_Wait(&rreq[ri], MPI_STATUS_IGNORE);

            pack_results(s, c, rb[ri]);

            MPI_Isend(
                rb[ri].data(),
                (int)rb[ri].size(),
                MPI_BYTE,
                0,
                TAG_RES,
                MPI_COMM_WORLD,
                &rreq[ri]
            );

            ri ^= 1;
        }

        MPI_Waitall(2, rreq, MPI_STATUSES_IGNORE);
    }

    // keep work buffers alive until the final sends complete
    if (rank == 0)
        MPI_Waitall(P, wreq.data(), MPI_STATUSES_IGNORE);
}