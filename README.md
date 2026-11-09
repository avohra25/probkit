# probkit

Probabilistic data structures in pure Python 3.10+, with no dependencies.

Work in progress. So far:

- **MurmurHash3** (x86_32 and x64_128), bit-compatible with the reference C++ code.
- **BloomFilter** with optimal sizing, double hashing, union/intersection,
  cardinality estimation and a compact binary format.
- **CountingBloomFilter** with 8-bit saturating counters, so keys can be removed.
- **CuckooFilter** using partial-key cuckoo hashing, with deletion, a victim
  slot so a full table never drops keys, and a binary format.
- **HyperLogLog** distinct counting with linear-counting fallback, merge,
  intersection and Jaccard estimates.
- **CountMinSketch** frequency estimation with optional conservative update,
  merge and inner-product (join size) estimates.
- **SpaceSaving** top-k / heavy hitters with per-item error bounds and merging.

```python
from probkit import BloomFilter

seen = BloomFilter(capacity=100_000, fpr=0.001)
seen.add("https://example.com/")
"https://example.com/" in seen   # True
"https://example.org/" in seen   # almost certainly False
```

Run the tests with `python -m unittest discover -s tests -t .`

MIT licensed.
