"""Bloom filters: a classic bit-array filter and a counting variant.

Both use Kirsch-Mitzenmacher double hashing: the ``i``-th probe for a key is
``(h1 + i * h2) mod m`` where ``h1, h2`` are the two halves of a single
MurmurHash3_x64_128 call. This gives the same asymptotic false-positive rate
as ``k`` independent hashes at a fraction of the cost.
"""

from __future__ import annotations

import math
import struct
from typing import Iterable

from ._hash import hash_pair

__all__ = ["BloomFilter", "CountingBloomFilter", "optimal_params"]

_LN2 = math.log(2)


def optimal_params(capacity: int, fpr: float) -> tuple[int, int]:
    """Return ``(num_bits, num_hashes)`` for ``capacity`` keys at rate ``fpr``.

    ``m = -n ln p / (ln 2)^2`` and ``k = (m / n) ln 2``, rounded sensibly.
    """
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if not 0.0 < fpr < 1.0:
        raise ValueError("fpr must be in (0, 1)")
    m = math.ceil(-capacity * math.log(fpr) / (_LN2 * _LN2))
    k = max(1, round(m / capacity * _LN2))
    return m, k


def _probes(item, m: int, k: int, seed: int):
    h1, h2 = hash_pair(item, seed)
    # Force h2 odd so probes never collapse onto one slot when gcd(h2, m) > 1
    # for power-of-two m.
    h2 |= 1
    for i in range(k):
        yield (h1 + i * h2) % m


class BloomFilter:
    """A space-efficient set with false positives but no false negatives.

    >>> bf = BloomFilter(capacity=1000, fpr=0.01)
    >>> bf.add("apple")
    True
    >>> "apple" in bf, "pear" in bf
    (True, False)
    """

    _MAGIC = b"PKBF"
    _HEADER = struct.Struct("<4sBQIIQ")  # magic, version, m, k, seed, inserts

    def __init__(self, capacity: int = 1000, fpr: float = 0.01, *,
                 num_bits: int | None = None, num_hashes: int | None = None,
                 seed: int = 0):
        if num_bits is None or num_hashes is None:
            m, k = optimal_params(capacity, fpr)
            num_bits = num_bits or m
            num_hashes = num_hashes or k
        if num_bits <= 0 or num_hashes <= 0:
            raise ValueError("num_bits and num_hashes must be positive")
        self.num_bits = num_bits
        self.num_hashes = num_hashes
        self.seed = seed & 0xFFFFFFFF
        self.inserts = 0
        self._bits = bytearray((num_bits + 7) // 8)

    # -- core operations -------------------------------------------------
    def add(self, item) -> bool:
        """Insert ``item``; return True if it was (probably) not present."""
        new = False
        bits = self._bits
        for p in _probes(item, self.num_bits, self.num_hashes, self.seed):
            byte, mask = p >> 3, 1 << (p & 7)
            if not bits[byte] & mask:
                bits[byte] |= mask
                new = True
        self.inserts += 1
        return new

    def update(self, items: Iterable) -> None:
        for item in items:
            self.add(item)

    def __contains__(self, item) -> bool:
        bits = self._bits
        return all(bits[p >> 3] & (1 << (p & 7))
                   for p in _probes(item, self.num_bits, self.num_hashes, self.seed))

    # -- statistics ------------------------------------------------------
    def bit_count(self) -> int:
        return sum(bin(b).count("1") for b in self._bits)

    def fill_ratio(self) -> float:
        return self.bit_count() / self.num_bits

    def estimated_fpr(self) -> float:
        """False-positive rate implied by the current fill: ``fill ** k``."""
        return self.fill_ratio() ** self.num_hashes

    def estimate_count(self) -> float:
        """Swamidass-Baldi estimate of distinct items inserted so far."""
        x = self.bit_count()
        m, k = self.num_bits, self.num_hashes
        if x >= m:
            return float("inf")
        return -m / k * math.log(1 - x / m)

    def __len__(self) -> int:
        return round(self.estimate_count()) if self.bit_count() < self.num_bits else self.inserts

    # -- set algebra -----------------------------------------------------
    def _check_compatible(self, other: "BloomFilter") -> None:
        if not isinstance(other, BloomFilter):
            raise TypeError("can only combine with another BloomFilter")
        if (self.num_bits, self.num_hashes, self.seed) != \
                (other.num_bits, other.num_hashes, other.seed):
            raise ValueError("filters have different parameters")

    def copy(self) -> "BloomFilter":
        out = BloomFilter(num_bits=self.num_bits, num_hashes=self.num_hashes, seed=self.seed)
        out._bits[:] = self._bits
        out.inserts = self.inserts
        return out

    def __or__(self, other: "BloomFilter") -> "BloomFilter":
        """Union: exactly the filter you'd get by inserting both key sets."""
        self._check_compatible(other)
        out = self.copy()
        out._bits = bytearray(a | b for a, b in zip(self._bits, other._bits))
        out.inserts = self.inserts + other.inserts
        return out

    def __and__(self, other: "BloomFilter") -> "BloomFilter":
        """Intersection: a superset of the true intersection filter."""
        self._check_compatible(other)
        out = self.copy()
        out._bits = bytearray(a & b for a, b in zip(self._bits, other._bits))
        out.inserts = min(self.inserts, other.inserts)
        return out

    def __eq__(self, other) -> bool:
        return (isinstance(other, BloomFilter)
                and (self.num_bits, self.num_hashes, self.seed) ==
                (other.num_bits, other.num_hashes, other.seed)
                and self._bits == other._bits)

    __hash__ = None  # mutable

    # -- serialisation ---------------------------------------------------
    def to_bytes(self) -> bytes:
        head = self._HEADER.pack(self._MAGIC, 1, self.num_bits, self.num_hashes,
                                 self.seed, self.inserts)
        return head + bytes(self._bits)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BloomFilter":
        hs = cls._HEADER.size
        if len(data) < hs:
            raise ValueError("truncated Bloom filter")
        magic, ver, m, k, seed, inserts = cls._HEADER.unpack_from(data)
        if magic != cls._MAGIC or ver != 1:
            raise ValueError("not a probkit Bloom filter")
        body = data[hs:]
        if len(body) != (m + 7) // 8:
            raise ValueError("Bloom filter body has wrong length")
        out = cls(num_bits=m, num_hashes=k, seed=seed)
        out._bits[:] = body
        out.inserts = inserts
        return out

    def __repr__(self) -> str:
        return (f"BloomFilter(num_bits={self.num_bits}, num_hashes={self.num_hashes}, "
                f"inserts={self.inserts})")


class CountingBloomFilter:
    """A Bloom filter with 8-bit saturating counters, so keys can be removed.

    Counters stick at 255 once saturated (they are never decremented again),
    which keeps the no-false-negatives guarantee intact.
    """

    MAX = 255

    def __init__(self, capacity: int = 1000, fpr: float = 0.01, *, seed: int = 0):
        self.num_counters, self.num_hashes = optimal_params(capacity, fpr)
        self.seed = seed & 0xFFFFFFFF
        self._c = bytearray(self.num_counters)
        self.size = 0

    def _slots(self, item):
        # Deduplicate so a key whose probes collide is counted once per slot.
        return set(_probes(item, self.num_counters, self.num_hashes, self.seed))

    def add(self, item) -> None:
        c = self._c
        for p in self._slots(item):
            if c[p] < self.MAX:
                c[p] += 1
        self.size += 1

    def remove(self, item) -> None:
        """Remove one occurrence; raises KeyError if ``item`` is surely absent."""
        slots = self._slots(item)
        c = self._c
        if not all(c[p] for p in slots):
            raise KeyError(item)
        for p in slots:
            if c[p] < self.MAX:
                c[p] -= 1
        self.size -= 1

    def discard(self, item) -> bool:
        try:
            self.remove(item)
            return True
        except KeyError:
            return False

    def __contains__(self, item) -> bool:
        c = self._c
        return all(c[p] for p in _probes(item, self.num_counters, self.num_hashes, self.seed))

    def count(self, item) -> int:
        """Upper bound on how many times ``item`` was added (min counter)."""
        return min(self._c[p] for p in self._slots(item))

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (f"CountingBloomFilter(num_counters={self.num_counters}, "
                f"num_hashes={self.num_hashes}, size={self.size})")
