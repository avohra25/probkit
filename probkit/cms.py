"""Count-Min sketch (Cormode and Muthukrishnan 2005).

A ``depth x width`` table of counters. With ``width = ceil(e / epsilon)`` and
``depth = ceil(ln(1 / delta))`` every point estimate satisfies
``true <= estimate <= true + epsilon * N`` with probability ``1 - delta``,
where ``N`` is the total count added.
"""

from __future__ import annotations

import math
import struct

from ._hash import hash_pair

__all__ = ["CountMinSketch"]


class CountMinSketch:
    _MAGIC = b"PKCM"
    _HEADER = struct.Struct("<4sBIIIBq")

    def __init__(self, epsilon: float = 0.001, delta: float = 0.01, *,
                 width: int | None = None, depth: int | None = None,
                 conservative: bool = False, seed: int = 0):
        if width is None or depth is None:
            if not 0 < epsilon < 1 or not 0 < delta < 1:
                raise ValueError("epsilon and delta must be in (0, 1)")
            width = width or math.ceil(math.e / epsilon)
            depth = depth or math.ceil(math.log(1 / delta))
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive")
        self.width = width
        self.depth = depth
        self.conservative = conservative
        self.seed = seed & 0xFFFFFFFF
        self.total = 0
        self._rows = [[0] * width for _ in range(depth)]

    @property
    def epsilon(self) -> float:
        return math.e / self.width

    @property
    def delta(self) -> float:
        return math.exp(-self.depth)

    def _cols(self, item) -> list[int]:
        h1, h2 = hash_pair(item, self.seed)
        h2 |= 1
        w = self.width
        return [(h1 + i * h2) % w for i in range(self.depth)]

    def add(self, item, count: int = 1) -> int:
        """Add ``count`` (>= 0) occurrences; return the new estimate."""
        if count < 0:
            raise ValueError("Count-Min sketch only supports non-negative counts")
        cols = self._cols(item)
        rows = self._rows
        self.total += count
        if self.conservative:
            # Conservative update: raise each counter only as far as needed.
            target = min(rows[i][c] for i, c in enumerate(cols)) + count
            for i, c in enumerate(cols):
                if rows[i][c] < target:
                    rows[i][c] = target
            return target
        est = None
        for i, c in enumerate(cols):
            rows[i][c] += count
            v = rows[i][c]
            est = v if est is None or v < est else est
        return est

    def __getitem__(self, item) -> int:
        return self.estimate(item)

    def estimate(self, item) -> int:
        rows = self._rows
        return min(rows[i][c] for i, c in enumerate(self._cols(item)))

    def inner_product(self, other: "CountMinSketch") -> int:
        """Upper-bound estimate of ``sum_x f(x) * g(x)`` (e.g. join size)."""
        self._check(other)
        return min(sum(a * b for a, b in zip(r1, r2))
                   for r1, r2 in zip(self._rows, other._rows))

    def merge(self, other: "CountMinSketch") -> None:
        """Add another sketch's counts (valid for both update modes)."""
        self._check(other)
        for r1, r2 in zip(self._rows, other._rows):
            for j, v in enumerate(r2):
                r1[j] += v
        self.total += other.total

    def _check(self, other) -> None:
        if not isinstance(other, CountMinSketch):
            raise TypeError("can only combine with another CountMinSketch")
        if (self.width, self.depth, self.seed) != (other.width, other.depth, other.seed):
            raise ValueError("sketches have different dimensions or seed")

    def to_bytes(self) -> bytes:
        head = self._HEADER.pack(self._MAGIC, 1, self.width, self.depth, self.seed,
                                 1 if self.conservative else 0, self.total)
        body = b"".join(struct.pack(f"<{self.width}Q", *row) for row in self._rows)
        return head + body

    @classmethod
    def from_bytes(cls, data: bytes) -> "CountMinSketch":
        hs = cls._HEADER.size
        if len(data) < hs:
            raise ValueError("truncated Count-Min sketch")
        magic, ver, w, d, seed, cons, total = cls._HEADER.unpack_from(data)
        if magic != cls._MAGIC or ver != 1 or w == 0 or d == 0:
            raise ValueError("not a probkit Count-Min sketch")
        if len(data) - hs != w * d * 8:
            raise ValueError("Count-Min body has wrong length")
        out = cls(width=w, depth=d, conservative=bool(cons), seed=seed)
        out.total = total
        out._rows = [list(struct.unpack_from(f"<{w}Q", data, hs + i * w * 8)) for i in range(d)]
        return out

    def __eq__(self, other) -> bool:
        return isinstance(other, CountMinSketch) and self.to_bytes() == other.to_bytes()

    __hash__ = None

    def __repr__(self) -> str:
        mode = ", conservative" if self.conservative else ""
        return f"CountMinSketch(width={self.width}, depth={self.depth}{mode}, total={self.total})"
