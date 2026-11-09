"""HyperLogLog cardinality estimation (Flajolet et al. 2007).

Uses a 64-bit hash, so no large-range correction is needed, and falls back
to linear counting for small cardinalities as in Heule et al. (HLL++).
Expected relative error is about ``1.04 / sqrt(2**precision)``.
"""

from __future__ import annotations

import math
import struct
from typing import Iterable

from ._hash import hash_pair

__all__ = ["HyperLogLog"]

_M64 = 0xFFFFFFFFFFFFFFFF


def _alpha(m: int) -> float:
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1 + 1.079 / m)


class HyperLogLog:
    """Estimate the number of distinct items in a stream in ``2**p`` bytes.

    >>> h = HyperLogLog(precision=12)
    >>> h.update(range(100_000))
    >>> abs(h.count() - 100_000) < 5000
    True
    """

    _MAGIC = b"PKHL"
    _HEADER = struct.Struct("<4sBBI")

    def __init__(self, precision: int = 14, *, seed: int = 0):
        if not 4 <= precision <= 18:
            raise ValueError("precision must be in [4, 18]")
        self.precision = precision
        self.m = 1 << precision
        self.seed = seed & 0xFFFFFFFF
        self._reg = bytearray(self.m)

    @property
    def relative_error(self) -> float:
        return 1.04 / math.sqrt(self.m)

    def add(self, item) -> bool:
        """Add ``item``; return True if a register changed."""
        h = hash_pair(item, self.seed)[0]
        return self._add_hash(h)

    def _add_hash(self, h: int) -> bool:
        p = self.precision
        idx = h >> (64 - p)
        rest = h & ((1 << (64 - p)) - 1)
        rho = (64 - p) - rest.bit_length() + 1  # position of the leftmost 1
        if rho > self._reg[idx]:
            self._reg[idx] = rho
            return True
        return False

    def update(self, items: Iterable) -> None:
        for item in items:
            self.add(item)

    def count(self) -> float:
        m = self.m
        z = 0.0
        zeros = 0
        for r in self._reg:
            z += 2.0 ** -r
            if r == 0:
                zeros += 1
        est = _alpha(m) * m * m / z
        if est <= 2.5 * m and zeros:
            return m * math.log(m / zeros)  # linear counting
        return est

    def __len__(self) -> int:
        return round(self.count())

    def merge(self, other: "HyperLogLog") -> None:
        """In-place union: the result estimates ``|A ∪ B|``."""
        self._check(other)
        self._reg = bytearray(max(a, b) for a, b in zip(self._reg, other._reg))

    def __or__(self, other: "HyperLogLog") -> "HyperLogLog":
        out = self.copy()
        out.merge(other)
        return out

    def intersection_estimate(self, other: "HyperLogLog") -> float:
        """Inclusion-exclusion estimate of ``|A ∩ B|`` (noisy when small)."""
        return max(0.0, self.count() + other.count() - (self | other).count())

    def jaccard_estimate(self, other: "HyperLogLog") -> float:
        union = (self | other).count()
        return self.intersection_estimate(other) / union if union else 0.0

    def _check(self, other) -> None:
        if not isinstance(other, HyperLogLog):
            raise TypeError("can only merge with another HyperLogLog")
        if (self.precision, self.seed) != (other.precision, other.seed):
            raise ValueError("sketches have different precision or seed")

    def copy(self) -> "HyperLogLog":
        out = HyperLogLog(self.precision, seed=self.seed)
        out._reg[:] = self._reg
        return out

    def __eq__(self, other) -> bool:
        return (isinstance(other, HyperLogLog) and self.precision == other.precision
                and self.seed == other.seed and self._reg == other._reg)

    __hash__ = None

    def to_bytes(self) -> bytes:
        return self._HEADER.pack(self._MAGIC, 1, self.precision, self.seed) + bytes(self._reg)

    @classmethod
    def from_bytes(cls, data: bytes) -> "HyperLogLog":
        hs = cls._HEADER.size
        if len(data) < hs:
            raise ValueError("truncated HyperLogLog")
        magic, ver, p, seed = cls._HEADER.unpack_from(data)
        if magic != cls._MAGIC or ver != 1 or not 4 <= p <= 18:
            raise ValueError("not a probkit HyperLogLog")
        if len(data) - hs != 1 << p:
            raise ValueError("HyperLogLog body has wrong length")
        if max(data[hs:], default=0) > 64 - p + 1:
            raise ValueError("HyperLogLog register out of range")
        out = cls(p, seed=seed)
        out._reg[:] = data[hs:]
        return out

    def __repr__(self) -> str:
        return f"HyperLogLog(precision={self.precision}, estimate={self.count():.0f})"
