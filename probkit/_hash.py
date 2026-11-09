"""Pure-Python MurmurHash3 (x86_32 and x64_128), plus key normalisation.

These are bit-for-bit compatible with Austin Appleby's reference
implementation (and therefore with the ``mmh3`` package), so sketches built
with probkit hash the same way as sketches built elsewhere.
"""

from __future__ import annotations

__all__ = ["murmur3_32", "murmur3_128", "key_bytes", "hash_pair"]

_M32 = 0xFFFFFFFF
_M64 = 0xFFFFFFFFFFFFFFFF


def key_bytes(item) -> bytes:
    """Turn a supported key into bytes.

    ``bytes``-like objects are used as is, ``str`` is UTF-8 encoded and
    ``int`` uses its decimal text form (so ``42`` and ``"42"`` collide, which
    is deliberate: keys usually come from text). Anything else is rejected
    rather than silently hashed through ``repr``.
    """
    if isinstance(item, (bytes, bytearray, memoryview)):
        return bytes(item)
    if isinstance(item, str):
        return item.encode("utf-8")
    if isinstance(item, int) and not isinstance(item, bool):
        return str(item).encode("ascii")
    raise TypeError(f"unsupported key type: {type(item).__name__}")


def _rotl32(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & _M32


def _rotl64(x: int, r: int) -> int:
    return ((x << r) | (x >> (64 - r))) & _M64


def _fmix32(h: int) -> int:
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & _M32
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & _M32
    h ^= h >> 16
    return h


def _fmix64(k: int) -> int:
    k ^= k >> 33
    k = (k * 0xFF51AFD7ED558CCD) & _M64
    k ^= k >> 33
    k = (k * 0xC4CEB9FE1A85EC53) & _M64
    k ^= k >> 33
    return k


def murmur3_32(data: bytes, seed: int = 0) -> int:
    """MurmurHash3_x86_32 as an unsigned 32-bit int."""
    c1, c2 = 0xCC9E2D51, 0x1B873593
    n = len(data)
    h = seed & _M32
    nblocks = n // 4
    for i in range(nblocks):
        k = int.from_bytes(data[4 * i:4 * i + 4], "little")
        k = (k * c1) & _M32
        k = _rotl32(k, 15)
        k = (k * c2) & _M32
        h ^= k
        h = _rotl32(h, 13)
        h = (h * 5 + 0xE6546B64) & _M32
    tail = data[4 * nblocks:]
    if tail:
        k = int.from_bytes(tail, "little")
        k = (k * c1) & _M32
        k = _rotl32(k, 15)
        k = (k * c2) & _M32
        h ^= k
    h ^= n
    return _fmix32(h)


def murmur3_128(data: bytes, seed: int = 0) -> tuple[int, int]:
    """MurmurHash3_x64_128 as two unsigned 64-bit halves ``(h1, h2)``."""
    c1, c2 = 0x87C37B91114253D5, 0x4CF5AD432745937F
    n = len(data)
    h1 = h2 = seed & _M32
    nblocks = n // 16
    for i in range(nblocks):
        off = 16 * i
        k1 = int.from_bytes(data[off:off + 8], "little")
        k2 = int.from_bytes(data[off + 8:off + 16], "little")
        k1 = (k1 * c1) & _M64
        k1 = _rotl64(k1, 31)
        k1 = (k1 * c2) & _M64
        h1 ^= k1
        h1 = _rotl64(h1, 27)
        h1 = (h1 + h2) & _M64
        h1 = (h1 * 5 + 0x52DCE729) & _M64
        k2 = (k2 * c2) & _M64
        k2 = _rotl64(k2, 33)
        k2 = (k2 * c1) & _M64
        h2 ^= k2
        h2 = _rotl64(h2, 31)
        h2 = (h2 + h1) & _M64
        h2 = (h2 * 5 + 0x38495AB5) & _M64
    tail = data[16 * nblocks:]
    if len(tail) > 8:
        k2 = int.from_bytes(tail[8:], "little")
        k2 = (k2 * c2) & _M64
        k2 = _rotl64(k2, 33)
        k2 = (k2 * c1) & _M64
        h2 ^= k2
    if tail:
        k1 = int.from_bytes(tail[:8], "little")
        k1 = (k1 * c1) & _M64
        k1 = _rotl64(k1, 31)
        k1 = (k1 * c2) & _M64
        h1 ^= k1
    h1 ^= n
    h2 ^= n
    h1 = (h1 + h2) & _M64
    h2 = (h2 + h1) & _M64
    h1 = _fmix64(h1)
    h2 = _fmix64(h2)
    h1 = (h1 + h2) & _M64
    h2 = (h2 + h1) & _M64
    return h1, h2


def hash_pair(item, seed: int = 0) -> tuple[int, int]:
    """Two independent-looking 64-bit hashes of a key (for double hashing)."""
    return murmur3_128(key_bytes(item), seed)
