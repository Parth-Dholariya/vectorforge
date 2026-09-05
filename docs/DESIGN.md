# VectorForge — Design Notes

This document records the decisions and, more importantly, the ones deliberately
*not* taken in v1. Where a simpler option was chosen over a more impressive one,
the reasoning is written down rather than left implicit.

---

## 1. Why HNSW

The candidates for the node-level index were IVF-Flat, IVF-PQ, and HNSW.

| | build | query | recall ceiling | updates |
|---|---|---|---|---|
| IVF-Flat | fast (k-means) | good with high `nprobe` | high | cheap inserts |
| IVF-PQ | fast | fastest, tiny memory | capped by quantisation error | cheap |
| HNSW | slow | best recall/latency tradeoff | ~1.0 | inserts fine, **deletes hard** |

HNSW wins on the axis that matters for a serving system: it gives the best
recall at a fixed latency budget, and latency is what a query path is judged on.
Its two costs are a slow build and no native delete. Both are real, and Phase 2
is largely about paying them down.

The deeper reason is that HNSW's `ef` parameter is a *runtime* knob. IVF's
`nprobe` is too, but IVF's recall ceiling is set at training time by the
centroid count. With HNSW a single built index can be moved along the entire
recall/latency curve without rebuilding, which is what lets a coordinator trade
recall for tail latency under load. That's a distributed-systems property, not
an ANN property, and it's why HNSW is the right base for this project
specifically.

## 2. Why not wrap FAISS

FAISS is the reference implementation, and using it would make the index
trivially good. It is used here **only as the benchmark baseline**, never as a
dependency of the index itself. The point of the project is the graph, the
memory layout, and the distribution — wrapping FAISS would leave nothing but
RPC plumbing.

## 3. Distance kernels: runtime dispatch, not `-march=native`

The obvious move is to compile everything with `-mavx2 -march=native`. Rejected:
a binary built that way crashes with SIGILL on any host without AVX2, and
`-march=native` bakes in the *build* machine's feature set, which is exactly
wrong for something that ships in a container to heterogeneous EKS nodes.

Instead each kernel carries `__attribute__((target("avx2,fma")))` and a
`__builtin_cpu_supports` check picks the implementation once at startup. One
binary, correct everywhere, full speed where the hardware allows.

A subtlety that made this non-optional: with a global `-mavx2`, GCC is free to
emit VEX-encoded instructions into the *dispatcher itself*, so the process
faults before it can ever run the feature test. The target-attribute approach
confines AVX2 to functions that are only reachable after the check.

**Four accumulators, not one.** The inner loop is a dependent chain of FMAs, and
an FMA has ~4 cycles of latency against a 0.5-cycle throughput on Zen+. A single
accumulator stalls on that chain and wastes ~7/8 of the FMA issue capacity; four
independent accumulators keep the pipes fed.

## 4. Cosine collapses to inner product

`normalize()` runs once per vector on insert. After that, cosine similarity is
exactly a dot product, so the search loop runs the inner-product kernel with no
extra work. The alternative — computing both norms inside the distance function
— adds two reductions to *every* distance evaluation, of which there are
millions per query batch, to save one reduction per vector at insert time.

This is a genuine API footgun and it bit during development:
`get_distance_fn(Metric::Cosine)` returns a function that is **not** cosine
distance unless its inputs are already unit length. The first version of the
recall test used it as a brute-force reference on raw vectors, which silently
ranked by inner product (magnitude-dominated on Gaussian data) and reported
0.671 recall against a true-cosine index that was in fact correct. The test now
computes ground truth in double precision from the metric's definition and
shares no code with the kernels it checks — a reference that reuses the
implementation's own kernel can only prove the implementation agrees with
itself.

## 5. Memory layout

Layer 0 is one contiguous arena, `stride_` bytes per element:

```
[ uint32 degree | idx_t neighbours[M0] | pad to 32B | float vector[dim] | label_t ]
```

Two properties are being bought here:

1. **One cache walk per node visit.** The search reads a node's neighbour list
   and then its vector. Interleaved, that is sequential; as separate heap
   allocations it is two pointer chases into unrelated pages.
2. **Phase 2 gets mmap almost for free.** The on-disk format *is* this arena, so
   loading an index is `mmap()` and setting a pointer — no deserialisation pass,
   no rebuild, and the page cache handles residency. This is the whole reason
   the layout was fixed before persistence was written.

The vector is padded to a 32-byte boundary. The kernels use unaligned loads so
this isn't required for correctness, but it keeps a 256-bit load from straddling
a cache line in the hottest loop in the system.

Layers ≥ 1 hold roughly `1/M` of the elements per level, so a dense arena would
be ~97% empty. Those get individual allocations.

## 6. Concurrency in v1

- **Concurrent `add`** — supported. Link lists are guarded by 65,536 striped
  mutexes rather than one mutex per element; at 10M elements per-element mutexes
  would cost more memory than the graph. Inserts that raise the top layer take a
  global lock, but those occur with probability ~`1/M` per level, so it is
  nearly uncontended.
- **Concurrent `search`** — supported, and lock-free. `greedy_descend` and
  `search_layer` are templated on a `Locking` bool so the query path takes no
  mutexes at all; paying a lock per graph hop on the read path would cost real
  QPS to defend against a writer that v1 says isn't there.
- **`search` concurrent with `add`** — **not supported.** Ingest and serving are
  separated at the shard level instead (Phase 3). Making this safe means either
  locking the read path, which costs the QPS the whole layout exists to buy, or
  epoch-based reclamation, which is a lot of machinery for a v1. Writing the
  restriction down is the honest option.

The multithreaded build test is really a race detector: a torn link list shows
up as a shredded graph, and recall collapses long before anything crashes.

## 7. Why not Raft in v1

Full Raft per shard would give linearizable writes and automatic leader
election. It is deliberately out of scope, because the workload does not pay for
it:

- Vector search is **read-dominated** — an embedding corpus is written once and
  queried continuously.
- The consistency requirement is weak. A vector missing from an index for a few
  hundred milliseconds after insert changes recall by ~1/n. There is no
  read-modify-write, no transaction, no invariant across keys for consensus to
  protect.
- Raft's cost is real: a write cannot ack until a quorum has it on disk, which
  puts a full RTT plus an fsync in the ingest path.

v1 uses primary-replica with async replication and a WAL. The failure mode is
explicit: if a primary dies before its replica has caught up, unreplicated
writes are lost, bounded by the replication lag, and the WAL bounds what a
restarted node loses to the last fsync batch. That is the correct trade for this
workload, and stating the lost-write window plainly is more useful than claiming
a guarantee the system doesn't make.

The place Raft *would* earn its keep is shard-placement metadata — the mapping
of shards to nodes, where a split brain genuinely corrupts the cluster. That is
a small amount of rarely-changing state, and it is where consensus goes if v2
needs it.

## 8. Deletion (Phase 2)

HNSW has no delete. Removing a node breaks the graph's navigability: its
neighbours lose an edge each, and if it was an entry point the layer above is
orphaned.

The plan is tombstones plus background compaction. A deleted element is marked
in a bitmap; searches over-fetch and filter, so a deleted node still *routes*
traffic even though it is never returned. Recall degrades as tombstones
accumulate, so compaction rebuilds a segment once its tombstone ratio crosses a
threshold. This puts the cost on a background thread instead of the query path,
which is the same reason LSM trees compact rather than delete in place.

---

## Open questions

- ~~Does the pruning heuristic under-fill neighbour lists on high-intrinsic-dimension
  data?~~ **Resolved.** Recall on random Gaussian-128 is weak (0.57 @ ef=100),
  but SIFT-1M at the same dimension and 10× the size reaches 0.9829 @ ef=100.
  An under-connected graph would degrade both; only the unstructured data
  degrades. The Gaussian result is the dataset, not the implementation.
- Build is 1,407 vec/s single-threaded — 12 minutes for SIFT-1M. `add` is
  already thread-safe, so a parallel build should be close to linear in cores.
  Worth measuring before Phase 3, since distributed build time is a real
  operational cost.
- `ef` is currently caller-supplied. Phase 5 wants it adaptive to queue depth.
