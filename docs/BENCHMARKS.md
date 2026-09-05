# Benchmarks

All numbers below are measured, not estimated. Anything not yet run is marked
as such rather than filled in with a plausible figure.

## Machine

| | |
|---|---|
| CPU | AMD Ryzen 7 3750H (Zen+, 4C/8T, AVX2) |
| RAM | 7.4 GB |
| Compiler | GCC 16.1.0, `-O3`, mingw-w64 UCRT |
| SIMD path | `avx2+fma` (confirmed by the dispatcher at runtime) |
| Threads | build and query both **single-threaded** unless stated |

This is a laptop, not a benchmarking rig. Absolute QPS is therefore a floor, and
the numbers are useful mainly as a *curve* — recall against latency — and for
relative comparison against a baseline measured on the same machine.

`memory_usage()` reports the full pre-allocated layer-0 arena, so it reflects
`max_elements`, not the live element count.

---

## Reproducing

```bash
./build/bench_recall --synthetic --n 100000 --dim 128 --nq 1000 --k 10

./build/bench_recall \
    --base  data/sift/sift_base.fvecs \
    --query data/sift/sift_query.fvecs \
    --gt    data/sift/sift_groundtruth.ivecs
```

Each `ef` row runs 100 untimed warm-up queries first, so the timings measure
search work rather than cold page faults on the arena.

---

## Synthetic Gaussian — the hard case

Random `N(0,1)` vectors. **This is close to the worst case for any graph index**
and is included precisely because it is unflattering: random data has intrinsic
dimensionality equal to its ambient dimension, so distances concentrate and
there is no structure to exploit. Real embeddings have far lower intrinsic
dimension, which is the reason ANN search works at all.

### n = 5,000 · d = 64 · M = 16 · efConstruction = 200

Build: 0.6 s (8,283 vec/s)

| ef | recall@10 | QPS | p99 (ms) |
|---:|---:|---:|---:|
| 10 | 0.5194 | 83,168 | 0.024 |
| 16 | 0.6402 | 74,114 | 0.017 |
| 32 | 0.8042 | 35,209 | 0.060 |
| 64 | 0.9350 | 20,631 | 0.101 |
| 100 | 0.9762 | 10,114 | 0.281 |
| 200 | 0.9972 | 6,166 | 0.466 |
| 400 | 0.9996 | 4,603 | 0.501 |
| 800 | 0.9998 | 2,897 | 0.708 |

This configuration double-checks the harness: the unit suite independently
measures 0.9730 at ef=100 on the same parameters with different seeds, and the
two agree.

### n = 100,000 · d = 128 · M = 16 · efConstruction = 200

Build: 68.7 s (1,455 vec/s), 67.6 MB, 3 levels

| ef | recall@10 | QPS | p99 (ms) |
|---:|---:|---:|---:|
| 10 | 0.1564 | 15,612 | 0.133 |
| 16 | 0.2170 | 11,822 | 0.138 |
| 32 | 0.3305 | 6,818 | 0.268 |
| 64 | 0.4689 | 3,851 | 0.402 |
| 100 | 0.5746 | 2,583 | 0.640 |
| 200 | 0.7306 | 1,332 | 1.244 |
| 400 | 0.8492 | 749 | 2.062 |
| 800 | 0.9274 | 427 | 3.186 |

Recall is much lower than at d=64 for the reason above — both `n` and the
intrinsic dimension went up. Reaching 0.93 requires `ef=800`, i.e. touching
~1% of the corpus per query. The contrast with SIFT below is the point of
including this table.

---

## SIFT-1M

1M × 128 SIFT descriptors, 10,000 queries, **published ground truth** (not
self-computed). The standard comparison point, and the number that means
something.

`M = 16`, `efConstruction = 200`. Build: 710.9 s single-threaded (1,407 vec/s),
675.7 MB arena, 5 levels.

| ef | recall@10 | QPS | p99 (ms) |
|---:|---:|---:|---:|
| 10 | 0.7083 | 9,637 | 0.320 |
| 16 | 0.7997 | 7,934 | 0.374 |
| 32 | 0.9035 | 5,481 | 0.423 |
| 64 | **0.9635** | **3,490** | 0.630 |
| 100 | **0.9829** | **2,545** | 0.676 |
| 200 | 0.9957 | 1,386 | 1.339 |
| 400 | 0.9988 | 792 | 2.112 |
| 800 | 0.9994 | 393 | 5.227 |

### Reading this against the Gaussian tables

Same dimension (128), 10× more vectors, and recall at `ef=100` goes from 0.5746
to 0.9829. That gap is the entire argument for why ANN search works in practice:
SIFT descriptors have an intrinsic dimensionality around 10–15 despite living in
128 dimensions, so the graph has real structure to exploit. Random Gaussian data
has none.

It also resolves the open question flagged in DESIGN.md — the weak Gaussian
numbers were the dataset, not an under-connected graph. If the pruning heuristic
were under-filling neighbour lists, SIFT would degrade too. It doesn't.

The knee of the curve is around `ef = 32–64`: 0.90 → 0.96 recall costs about
36% of the QPS, while pushing 0.96 → 0.9994 costs another 89%. For a serving
system that is the useful region, and it is reachable per-query at runtime
without touching the built index.

---

## FAISS baseline

> **Not yet run.** To be measured on the same machine, same dataset, same `M`
> and `efConstruction`, sweeping `efSearch` over the same grid. Comparing
> against published FAISS numbers from other hardware would not be a comparison.

---

## Known gaps

- Build is single-threaded in the benchmark. The index supports concurrent
  `add`; a parallel build should cut wall-clock substantially and is worth
  measuring separately from query performance.
- Layer-0 degree distribution is not yet instrumented. DESIGN.md flags this as
  an open question — whether the pruning heuristic under-fills neighbour lists
  on high-intrinsic-dimension data, or whether the weak Gaussian-128 numbers are
  purely the data.
- No memory-resident vs page-cache distinction yet; that becomes meaningful in
  Phase 2 once the index is mmap-backed.
