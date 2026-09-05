# Paper notes — HNSW

Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search
using Hierarchical Navigable Small World graphs*, TPAMI 2018 (arXiv 1603.09320).

Notes taken while implementing `core/src/hnsw.cpp`. Recorded here because the
things that were confusing on the first read are the things worth remembering.

---

## The core idea

Take a proximity graph, do greedy descent: from some entry point, repeatedly hop
to whichever neighbour is closest to the query, stop at a local minimum. This
works, and it fails in one specific way — greedy descent gets stuck in local
minima, and a graph with only short edges has many of them.

NSW's fix was to insert nodes in random order, which incidentally produces some
long edges early on. HNSW's fix is deliberate: put the long edges on their own
layers.

Layer *l* contains a random subset of layer *l−1*, sampled with probability
`1/M`. Search starts at the top (very few nodes, so edges span the whole space),
descends greedily to the local minimum on that layer, then drops down and
repeats with a denser graph. Each layer refines the previous one's answer.

This is a skip list where the linked list is replaced by a proximity graph. That
analogy is the fastest way to remember the whole structure — including why the
level distribution is geometric.

## Level assignment (eq. 1)

```
l = floor(-ln(unif(0,1)) * mL),  mL = 1/ln(M)
```

The exponential produces a geometric level distribution: `1/M` of the elements
reach layer 1, `1/M²` reach layer 2, and so on. Expected layers = `ln(N)/ln(M)`,
which is where the logarithmic search complexity comes from.

The paper notes `mL = 1/ln(M)` minimises the overlap between layers, and that
setting `mL = 0` (single layer) degrades HNSW to NSW. Worth knowing — the
hierarchy is exactly what buys the `log N` scaling over NSW's polylog.

## Algorithm 2 — SEARCH-LAYER

The part I got wrong on the first pass. Two priority queues, opposite orders:

- `frontier` — **min**-heap, unexplored candidates, pop the closest first
- `top` — **max**-heap, the `ef` best found, so the *worst* is at the top and
  can be evicted in O(log ef)

The termination condition is the subtle bit:

```
if (nearest_unexplored > worst_kept && top.size() >= ef) break;
```

Everything left to explore is further away than the worst result already held,
so no further exploration can improve the answer. Dropping the `top.size() >= ef`
guard makes the search terminate early before the result set is even full — a
bug that shows up as *low recall at low ef only*, which is easy to misread as
a tuning problem rather than a correctness one.

The visited set is per-search. Using a `vector<bool>` cleared per query costs
O(N) per query while a search only touches O(ef · M) nodes; at N = 1M the memset
dominates everything else. Generation stamping (`visited_list.hpp`) makes the
clear O(1).

## Algorithm 4 — SELECT-NEIGHBORS-HEURISTIC

The single most important detail in the paper, and the one that looks
counter-intuitive.

The naive choice is "keep the M nearest candidates." The heuristic instead keeps
a candidate `c` only if it is closer to the base element than to any neighbour
already kept:

```
for c in candidates (ascending by distance to base):
    if all(dist(c, r) >= dist(c, base) for r in kept):
        kept.append(c)
```

Why: if the base element sits at the edge of a cluster, its M nearest neighbours
are all *inside* that cluster, in nearly the same direction. Every one of those
edges is redundant, and the node has no edge pointing anywhere else — greedy
descent that arrives from outside the cluster can never leave. The heuristic
rejects a candidate that is "shadowed" by one already selected, which forces the
retained edges to spread out in direction.

This is what keeps the graph connected across clusters. §4.2 shows it is the
difference between working and not working on clustered data; on uniform random
data the two selection rules perform almost identically, which is a good
reminder that benchmarking only on random data would hide the bug entirely.

## Bidirectional links and pruning

Links are undirected, so connecting A→B also adds B→A. That back-edge can push B
over its degree budget, and when it does, B re-runs the same heuristic over
{existing neighbours} ∪ {A} and keeps the best spread. The pruning is *not*
"drop the furthest" — using the heuristic here is what preserves B's long-range
edges too.

`M0 = 2M` on layer 0. The paper's reasoning is that layer 0 holds every element
and carries all the final-stage traffic, so it can afford more edges; the memory
is dominated by layer 0 either way.

## Parameters

| | effect | cost |
|---|---|---|
| `M` | graph degree; higher → better recall, especially at high dimension | memory, build time |
| `efConstruction` | beam width at build | build time only; free at query time |
| `ef` | beam width at query | **runtime knob** — the entire recall/latency curve |

`M` 5–48 is the useful range per the paper; 16 is a good default. The key
practical property is that `ef` is tunable per-query on an already-built index,
which is what lets a serving layer trade recall for tail latency under load.

## Things the paper does not solve

- **Deletion.** Not addressed at all. Removing a node strands its neighbours'
  edges and can orphan a layer if it was an entry point. See DESIGN.md §8.
- **Concurrent insert.** The paper is single-threaded. Fine-grained locking of
  the link lists is an implementation concern, and the correctness bar is that
  a torn neighbour list must never be observed mid-rewrite.
- **Build cost.** Building is far more expensive than IVF's k-means. This is the
  real price of HNSW and the reason a distributed build matters at scale.

## What surprised me

1. The heuristic matters *more* than the hierarchy on clustered data. Removing
   the hierarchy costs a constant factor; removing the heuristic breaks
   connectivity outright.
2. Recall on random Gaussian data is dramatically worse than on SIFT at equal
   `n` and `d` — 0.57 vs SIFT's much higher figure at the same `ef`. Random data
   has intrinsic dimensionality equal to its ambient dimension, so distances
   concentrate and there is no structure for a graph to exploit. Real embeddings
   have low intrinsic dimension, which is *why* ANN works at all. This is worth
   stating plainly: benchmarking an ANN index on random vectors measures the
   near-worst case and understates real-world performance.
