# VectorForge

A vector database that has an HNSW index which is entirely newly developed.

The graph, the distance kernels, and the memory layout are implemented here —
FAISS is used as the benchmark baseline and not as a dependency. The aim is an
index that holds its own against FAISS-HNSW on a single node, and a distribution
A layer which shatters and replicates it.

**Status: Phase 1 (single-node index) is complete.** Please see [the roadmap](#roadmap).

### SIFT-1M, single node, single thread

1M times 128 descriptors, with the published ground truth, where M is 16 and efConstruction is 200.
The AMD Ryzen 7 3750H, with AVX2, includes the full curve and methodology
[BENCHMARKS.md](docs/BENCHMARKS.md).

| ef | recall@10 | QPS | p99 |
|---:|---:|---:|---:|
| 32 | 0.9035 | 5,481 | 0.42 ms |
| 64 | 0.9635 | 3,490 | 0.63 ms |
| 100 | 0.9829 | 2,545 | 0.68 ms |
| 200 | 0.9957 | 1,386 | 1.34 ms |

---

## Quickstart

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Benchmark on synthetic data (no download):

```bash
./build/bench_recall --synthetic --n 100000 --dim 128 --nq 1000 --k 10
```

Benchmark on SIFT-1M:

```bash
The command python scripts/fetch_datasets.py sift takes up about 161 MB.
./build/bench_recall \
    --base  data/sift/sift_base.fvecs \
    --query data/sift/sift_query.fvecs \
    --gt    data/sift/sift_groundtruth.ivecs
```

### Windows

The script env.ps1 fixes the 64-bit mingw-w64 toolchain — this being necessary because a
32-bit compiler on `PATH` will silently produce a binary that cannot address
sufficient memory to store a real index.

```powershell
winget install -e --id BrechtSanders.WinLibs.POSIX.UCRT
. .\scripts\env.ps1
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

---

## What is implemented

**HNSW index** (`core/`) — hierarchical layers with geometric level assignment,
beam search (Algorithm 2), the neighbour-selection heuristic (Algorithm 4), and
Links in both directions using a heuristic-based method when there is an overflow in degree. M,
Construction and ef are both provided; ef is a runtime option for each query.

**Distance kernels** — L2, inner product, and cosine, with AVX2+FMA
Implementations are selected at runtime by using `cpuid`. There are four separate accumulators.
Therefore the FMA latency chain does not serialise since cosine is normalised at the time of insertion and then
runs as a simple dot product.

**Concurrency** — multi-threaded `add` via striped link locks; lock-free
The v1 version does not support searching while items are being added.
(see [DESIGN.md](docs/DESIGN.md#6-concurrency-in-v1)).

**Tests** — recall is verified against exhaustive brute force computed in double
precision from each metric's definition, sharing no code with the kernels under
Test. The suite also enforces the monotonicity of `ef`, its self-retrieval, and the ordering of the results,
It carries out a multi-threaded build using a race detector.

---

## Docs

- **[DESIGN.md](docs/DESIGN.md)** — the reasons for choosing HNSW over IVF and the rationale for runtime SIMD dispatch
  the layer-0 memory layout, and why v1 deliberately has no Raft
- **[PAPER_NOTES.md](docs/PAPER_NOTES.md)**: a step-by-step account of Malkov & Yashunin's work
— the recall/QPS curves as specified in [BENCHMARKS.md](docs/BENCHMARKS.md))

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Single-node HNSW, SIMD kernels, recall vs brute force | **done** |
| 2 | mmap persistence, WAL, tombstone deletes + compaction | planned |
| 3 | gRPC, consistent-hash sharding, scatter-gather, replication | planned |
| 4 | EKS + Terraform, Prometheus/Grafana, chaos tests | planned |

The layer-0 arena was laid out with Phase 2 in mind: the on-disk format is the
in-memory format, so loading an index is an `mmap` and a pointer assignment
instead of a deserialisation pass.

## References

- Malkov & Yashunin, [*Efficient and robust approximate nearest neighbor search
  Using HNSW graphs [TPAMI 2018](https://arxiv.org/abs/1603.09320)
- [TEXMEX SIFT/GIST corpora](http://corpus-texmex.irisa.fr/)