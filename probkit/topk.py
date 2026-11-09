"""Space-Saving heavy hitters (Metwally, Agrawal, El Abbadi 2005).

Tracks at most ``capacity`` items. When a new item arrives and the table is
full, the item with the smallest count is evicted and the newcomer inherits
its count (recorded as the newcomer's maximum overestimation, ``error``).

Guarantees, with ``N`` the stream length:

* ``count - error <= true frequency <= count``
* every item with true frequency ``> N / capacity`` is in the table.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Hashable, Iterable

__all__ = ["SpaceSaving", "HeavyHitter"]


@dataclass(frozen=True)
class HeavyHitter:
    item: Hashable
    count: int
    error: int

    @property
    def guaranteed(self) -> int:
        """Lower bound on the true frequency."""
        return self.count - self.error


class SpaceSaving:
    def __init__(self, capacity: int = 100):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.total = 0
        self._count: dict = {}
        self._error: dict = {}
        # Lazy min-heap of (count, tiebreak, item); stale entries are skipped.
        self._heap: list = []
        self._tick = 0

    def _push(self, item) -> None:
        self._tick += 1
        heapq.heappush(self._heap, (self._count[item], self._tick, item))
        if len(self._heap) > 4 * self.capacity + 16:
            self._rebuild()

    def _rebuild(self) -> None:
        self._heap = [(c, i, it) for i, (it, c) in enumerate(self._count.items())]
        heapq.heapify(self._heap)
        self._tick = len(self._heap)

    def _pop_min(self):
        while True:
            c, _, item = heapq.heappop(self._heap)
            if self._count.get(item) == c:
                return item, c

    def add(self, item, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("count must be positive")
        self.total += count
        if item in self._count:
            self._count[item] += count
        elif len(self._count) < self.capacity:
            self._count[item] = count
            self._error[item] = 0
        else:
            victim, vcount = self._pop_min()
            del self._count[victim]
            del self._error[victim]
            self._count[item] = vcount + count
            self._error[item] = vcount
        self._push(item)

    def update(self, items: Iterable) -> None:
        for item in items:
            self.add(item)

    def __contains__(self, item) -> bool:
        return item in self._count

    def __len__(self) -> int:
        return len(self._count)

    def estimate(self, item) -> HeavyHitter | None:
        if item not in self._count:
            return None
        return HeavyHitter(item, self._count[item], self._error[item])

    def top(self, n: int | None = None) -> list[HeavyHitter]:
        """Tracked items by descending count (ties broken by lower error)."""
        hits = [HeavyHitter(k, c, self._error[k]) for k, c in self._count.items()]
        hits.sort(key=lambda h: (-h.count, h.error, str(h.item)))
        return hits if n is None else hits[:n]

    def heavy_hitters(self, phi: float) -> list[HeavyHitter]:
        """Candidates with frequency possibly ``> phi * N``.

        Complete (no false negatives) whenever ``phi >= 1 / capacity``.
        """
        if not 0 < phi <= 1:
            raise ValueError("phi must be in (0, 1]")
        threshold = phi * self.total
        return [h for h in self.top() if h.count > threshold]

    def merge(self, other: "SpaceSaving") -> None:
        """Combine two summaries (Agarwal et al. mergeable-summary rule)."""
        if not isinstance(other, SpaceSaving):
            raise TypeError("can only merge with another SpaceSaving")
        floor_a = min(self._count.values()) if len(self._count) >= self.capacity else 0
        floor_b = min(other._count.values()) if len(other._count) >= other.capacity else 0
        keys = set(self._count) | set(other._count)
        merged = {}
        for k in keys:
            ca = self._count.get(k)
            cb = other._count.get(k)
            ea = self._error.get(k, floor_a)
            eb = other._error.get(k, floor_b)
            merged[k] = ((ca if ca is not None else floor_a) + (cb if cb is not None else floor_b),
                         ea + eb)
        best = sorted(merged.items(), key=lambda kv: (-kv[1][0], kv[1][1], str(kv[0])))
        best = best[:self.capacity]
        self._count = {k: c for k, (c, _) in best}
        self._error = {k: e for k, (_, e) in best}
        self.total += other.total
        self._rebuild()

    def __repr__(self) -> str:
        return f"SpaceSaving(capacity={self.capacity}, tracked={len(self)}, total={self.total})"
