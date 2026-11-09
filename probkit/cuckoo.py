"""Cuckoo filter (Fan, Andersen, Kaminsky, Mitzenmacher 2014).

Stores short fingerprints in a table of 4-slot buckets. Each key has two
candidate buckets ``i1 = h & mask`` and ``i2 = i1 ^ (hash(fp) & mask)``; the
XOR trick ("partial-key cuckoo hashing") means either bucket can be computed
from the other plus the fingerprint, so entries can be relocated without the
original key. Unlike a Bloom filter it supports deletion, and at low target
error rates it is smaller.
"""

from __future__ import annotations

import math
import random
import struct

from ._hash import hash_pair, murmur3_32

__all__ = ["CuckooFilter", "CuckooFilterFull"]


class CuckooFilterFull(Exception):
    """Raised by :meth:`CuckooFilter.add` when no slot could be freed."""


class CuckooFilter:
    """Approximate set membership with deletion.

    ``fingerprint_bits`` controls the false-positive rate, which is bounded
    by roughly ``2 * bucket_size / 2**fingerprint_bits``.
    """

    _MAGIC = b"PKCF"
    _HEADER = struct.Struct("<4sBIBBIHIQ")

    def __init__(self, capacity: int = 1000, *, fingerprint_bits: int = 12,
                 bucket_size: int = 4, max_kicks: int = 500, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 4 <= fingerprint_bits <= 32:
            raise ValueError("fingerprint_bits must be in [4, 32]")
        if bucket_size < 1:
            raise ValueError("bucket_size must be positive")
        # Aim for ~95% maximum load; bucket count must be a power of two for
        # the XOR alternate-index trick to be an involution.
        wanted = math.ceil(capacity / (bucket_size * 0.95))
        self.num_buckets = 1 << max(1, (wanted - 1).bit_length())
        self.bucket_size = bucket_size
        self.fingerprint_bits = fingerprint_bits
        self.max_kicks = max_kicks
        self.seed = seed & 0xFFFFFFFF
        self._mask = self.num_buckets - 1
        self._buckets: list[list[int]] = [[] for _ in range(self.num_buckets)]
        self._victim: tuple[int, int] | None = None  # (index, fingerprint)
        self._rng = random.Random(self.seed)
        self.size = 0

    # -- hashing ---------------------------------------------------------
    def _fp_and_index(self, item) -> tuple[int, int]:
        h1, h2 = hash_pair(item, self.seed)
        fp = h2 & ((1 << self.fingerprint_bits) - 1)
        if fp == 0:  # 0 is never stored, so remap it to a valid fingerprint
            fp = 1
        return fp, h1 & self._mask

    def _alt(self, index: int, fp: int) -> int:
        return (index ^ murmur3_32(fp.to_bytes(4, "little"), self.seed)) & self._mask

    # -- operations ------------------------------------------------------
    def add(self, item) -> None:
        """Insert ``item`` (duplicates allowed, up to 2 * bucket_size copies).

        Raises :class:`CuckooFilterFull` if the table is too full; the filter
        stays consistent (no previously added key is lost) when that happens.
        """
        if self._victim is not None:
            raise CuckooFilterFull("filter is full")
        fp, i1 = self._fp_and_index(item)
        i2 = self._alt(i1, fp)
        for i in (i1, i2):
            if len(self._buckets[i]) < self.bucket_size:
                self._buckets[i].append(fp)
                self.size += 1
                return
        i = self._rng.choice((i1, i2))
        for _ in range(self.max_kicks):
            bucket = self._buckets[i]
            j = self._rng.randrange(len(bucket))
            fp, bucket[j] = bucket[j], fp
            i = self._alt(i, fp)
            if len(self._buckets[i]) < self.bucket_size:
                self._buckets[i].append(fp)
                self.size += 1
                return
        # Park the homeless fingerprint so lookups still find it.
        self._victim = (i, fp)
        self.size += 1
        raise CuckooFilterFull("filter is full")

    def try_add(self, item) -> bool:
        """Like :meth:`add` but returns False instead of raising when full.

        Note that a False result after the table fills may still have stored
        the key (as the parked victim); the key is never lost either way.
        """
        try:
            self.add(item)
            return True
        except CuckooFilterFull:
            return False

    def __contains__(self, item) -> bool:
        fp, i1 = self._fp_and_index(item)
        i2 = self._alt(i1, fp)
        if fp in self._buckets[i1] or fp in self._buckets[i2]:
            return True
        v = self._victim
        return v is not None and v[1] == fp and v[0] in (i1, i2)

    def remove(self, item) -> None:
        """Delete one copy of ``item``. Only delete keys you actually added."""
        fp, i1 = self._fp_and_index(item)
        i2 = self._alt(i1, fp)
        for i in (i1, i2):
            bucket = self._buckets[i]
            if fp in bucket:
                bucket.remove(fp)
                self.size -= 1
                self._reinsert_victim()
                return
        v = self._victim
        if v is not None and v[1] == fp and v[0] in (i1, i2):
            self._victim = None
            self.size -= 1
            return
        raise KeyError(item)

    def discard(self, item) -> bool:
        try:
            self.remove(item)
            return True
        except KeyError:
            return False

    def _reinsert_victim(self) -> None:
        if self._victim is None:
            return
        i, fp = self._victim
        self._victim = None
        self.size -= 1
        for idx in (i, self._alt(i, fp)):
            if len(self._buckets[idx]) < self.bucket_size:
                self._buckets[idx].append(fp)
                self.size += 1
                return
        self._victim = (i, fp)
        self.size += 1

    # -- statistics ------------------------------------------------------
    def __len__(self) -> int:
        return self.size

    @property
    def capacity(self) -> int:
        return self.num_buckets * self.bucket_size

    def load_factor(self) -> float:
        return self.size / self.capacity

    def fpr_bound(self) -> float:
        return min(1.0, 2 * self.bucket_size / 2 ** self.fingerprint_bits)

    # -- serialisation ---------------------------------------------------
    def to_bytes(self) -> bytes:
        vi, vf = self._victim if self._victim else (0, 0)
        head = self._HEADER.pack(self._MAGIC, 1, self.num_buckets, self.bucket_size,
                                 self.fingerprint_bits, self.seed,
                                 1 if self._victim else 0, vi, vf)
        body = bytearray()
        for bucket in self._buckets:
            padded = bucket + [0] * (self.bucket_size - len(bucket))
            for fp in padded:
                body += fp.to_bytes(4, "little")
        return head + bytes(body)

    @classmethod
    def from_bytes(cls, data: bytes) -> "CuckooFilter":
        hs = cls._HEADER.size
        if len(data) < hs:
            raise ValueError("truncated cuckoo filter")
        magic, ver, nb, bs, fbits, seed, has_v, vi, vf = cls._HEADER.unpack_from(data)
        if magic != cls._MAGIC or ver != 1 or nb & (nb - 1) or nb == 0:
            raise ValueError("not a probkit cuckoo filter")
        if len(data) - hs != nb * bs * 4:
            raise ValueError("cuckoo filter body has wrong length")
        out = cls(nb * bs, fingerprint_bits=fbits, bucket_size=bs, seed=seed)
        out.num_buckets, out._mask = nb, nb - 1
        out._buckets = []
        off = hs
        for _ in range(nb):
            bucket = []
            for _ in range(bs):
                fp = int.from_bytes(data[off:off + 4], "little")
                off += 4
                if fp:
                    bucket.append(fp)
            out._buckets.append(bucket)
        out._victim = (vi, vf) if has_v else None
        out.size = sum(map(len, out._buckets)) + (1 if has_v else 0)
        return out

    def __eq__(self, other) -> bool:
        return (isinstance(other, CuckooFilter)
                and self.to_bytes() == other.to_bytes())

    __hash__ = None

    def __repr__(self) -> str:
        return (f"CuckooFilter(buckets={self.num_buckets}x{self.bucket_size}, "
                f"fingerprint_bits={self.fingerprint_bits}, size={self.size})")
