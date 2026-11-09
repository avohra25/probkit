# probkit

Probabilistic data structures in pure Python 3.10+, with no dependencies.
They answer "have I seen this?", "how many distinct?", "how often?" and
"what's most common?" over large streams in small, fixed memory, with
error bounds you can state up front.

| Structure | Answers | Guarantee |
|---|---|---|
| `BloomFilter` | membership | no false negatives; FPR set at construction |
| `CountingBloomFilter` | membership with deletes | as above, 8-bit saturating counters |
| `CuckooFilter` | membership with deletes | FPR ≤ 2b / 2^f; never drops a key when full |
| `HyperLogLog` | distinct count | ~1.04/√m relative error, mergeable |
| `CountMinSketch` | per-item frequency | true ≤ est ≤ true + εN with prob. 1 − δ |
| `SpaceSaving` | top-k / heavy hitters | every item with freq > N/k is tracked, with error bounds |
| `Reservoir` | uniform sample | exact uniform k-of-N sample in one pass (Algorithm L) |

Everything hashes with a from-scratch MurmurHash3 (x86_32 and x64_128) that
is bit-for-bit compatible with the reference C++ implementation, so the
filters are deterministic across runs and machines (Python's built-in `hash`
is salted per process). Most structures serialise to a compact binary format
and can be merged with a compatible sketch.

## Install

```
pip install git+https://github.com/avohra25/probkit
```

or just copy the `probkit/` folder; it only uses the standard library.

## Examples

```python
from probkit import BloomFilter, CuckooFilter, HyperLogLog, CountMinSketch, SpaceSaving

# Membership: 100k URLs at 0.1% false positives in ~180 KB
seen = BloomFilter(capacity=100_000, fpr=0.001)
seen.add("https://example.com/")
"https://example.com/" in seen          # True
"https://example.org/" in seen          # almost certainly False
blob = seen.to_bytes()                  # BloomFilter.from_bytes(blob) restores it

# Membership with deletes
cf = CuckooFilter(capacity=10_000, fingerprint_bits=12)
cf.add("session-42"); cf.remove("session-42")

# Distinct users per day, then the union across days
mon, tue = HyperLogLog(14), HyperLogLog(14)   # 16 KB each, ~0.8% error
mon.update(user_ids_monday); tue.update(user_ids_tuesday)
(mon | tue).count(), mon.jaccard_estimate(tue)

# Frequencies with conservative update (tighter overestimates)
cms = CountMinSketch(epsilon=0.001, delta=0.01, conservative=True)
for word in words:
    cms.add(word)
cms["the"]                              # never less than the true count

# Heavy hitters
ss = SpaceSaving(capacity=1000)
ss.update(query_log)
for h in ss.top(10):
    print(h.item, h.count, "at least", h.guaranteed)
```

## CLI

Installing adds a `probkit` command (also runnable as `python -m probkit`).
All commands read lines from a file or from stdin. The session below is illustrative (the numbers are made up):

```
$ probkit count access.log            # distinct lines via HyperLogLog
lines        1204331
distinct     ~88412  (±0.8% at 1σ, 16384 registers)

$ cut -d' ' -f1 access.log | probkit top -k 3
41022  10.0.0.7
17310  10.0.0.12
 9877  192.168.1.40

$ probkit bloom build banned.txt -o banned.bloom --fpr 0.0001
$ probkit bloom check banned.bloom alice bob --strict

$ probkit sample huge.csv -k 1000 --seed 7   # uniform sample in one pass
$ probkit hash hello                         # MurmurHash3 digests
```

## How it's built

- **Hashing.** `murmur3_128` gives two 64-bit halves; Bloom and Count-Min use
  Kirsch-Mitzenmacher double hashing (`h1 + i·h2`, with `h2` forced odd) to
  derive all `k` probe positions from one hash call.
- **Bloom sizing.** `m = −n ln p / (ln 2)²`, `k = (m/n) ln 2`. `estimate_count`
  uses the Swamidass-Baldi formula on the number of set bits.
- **Cuckoo filter.** Power-of-two bucket count so the alternate index
  `i ⊕ hash(fp)` is an involution; four slots per bucket; random-walk eviction
  with a seeded RNG. When eviction gives up, the homeless fingerprint is
  parked in a victim slot (as in the reference implementation), so lookups
  still find it, and it is reinserted as soon as a delete frees room.
- **HyperLogLog.** 64-bit hashes (no large-range correction needed) with
  linear counting below 2.5m. Merge is register-wise max; intersection and
  Jaccard use inclusion-exclusion.
- **Count-Min.** Width `⌈e/ε⌉`, depth `⌈ln 1/δ⌉`; optional conservative
  update; merge and inner-product (join size) estimates.
- **Space-Saving.** Dict plus a lazy min-heap for O(log k) evictions; tracks
  each item's overestimation so `count − error ≤ true ≤ count`. Summaries
  merge using the mergeable-summaries rule.
- **Reservoir.** Algorithm L skips ahead geometrically, so it draws
  O(k log(N/k)) random numbers rather than one per item.

## Tests

```
python -m unittest discover -s tests -t .
```

The suite checks the hash against reference vectors, and checks the
statistical claims empirically on seeded data: Bloom and cuckoo
false-positive rates against their targets, HyperLogLog error within three
standard errors across precisions plus an unbiasedness check over seeds,
Count-Min bounds (and that conservative update is tighter) on Zipf streams,
Space-Saving's bounds and completeness, merge results equal to building
from the combined stream, a chi-square test for reservoir uniformity, binary
round-trips with corruption rejection, and the CLI end to end. It runs on Python 3.10 to 3.13.

## License

MIT
