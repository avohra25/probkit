"""Uniform reservoir sampling with Li's Algorithm L (1994).

Draws ``k`` items uniformly without replacement from a stream of unknown
length in one pass, using O(k (1 + log(N / k))) random numbers instead of one
per item, by jumping straight to the next item that enters the reservoir.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, TypeVar

__all__ = ["Reservoir", "reservoir_sample"]

T = TypeVar("T")


class Reservoir:
    def __init__(self, k: int, *, seed: int | None = None):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.seen = 0
        self.sample: list = []
        self._rng = random.Random(seed)
        self._w = 1.0
        self._next = 0  # index of the next item that will replace a slot

    def _u(self) -> float:
        # random() can return 0.0; log(0) is undefined, so redraw.
        while True:
            u = self._rng.random()
            if u > 0.0:
                return u

    def _schedule(self) -> None:
        self._w *= math.exp(math.log(self._u()) / self.k)
        self._next += math.floor(math.log(self._u()) / math.log1p(-self._w)) + 1

    def add(self, item) -> None:
        i = self.seen
        self.seen += 1
        if i < self.k:
            self.sample.append(item)
            if i == self.k - 1:
                self._next = self.k - 1
                self._schedule()
            return
        if i == self._next:
            self.sample[self._rng.randrange(self.k)] = item
            self._schedule()

    def update(self, items: Iterable) -> None:
        for item in items:
            self.add(item)

    def __len__(self) -> int:
        return len(self.sample)


def reservoir_sample(items: Iterable[T], k: int, *, seed: int | None = None) -> list[T]:
    r = Reservoir(k, seed=seed)
    r.update(items)
    return r.sample
