"""probkit: probabilistic data structures in pure Python."""

from ._hash import murmur3_128, murmur3_32
from .bloom import BloomFilter, CountingBloomFilter, optimal_params

__all__ = ["BloomFilter", "CountingBloomFilter", "optimal_params",
           "murmur3_32", "murmur3_128"]
__version__ = "0.1.0"
