import collections
import random
import unittest

from probkit.topk import SpaceSaving


def zipf_stream(n, universe, s=1.2, seed=0):
    rng = random.Random(seed)
    weights = [1 / (r ** s) for r in range(1, universe + 1)]
    return [f"item{x}" for x in rng.choices(range(universe), weights=weights, k=n)]


class SpaceSavingTest(unittest.TestCase):
    def check_invariants(self, ss, truth):
        self.assertLessEqual(len(ss), ss.capacity)
        # Evictions hand their count to the newcomer, so counts sum to N.
        self.assertEqual(sum(h.count for h in ss.top()), ss.total)
        for h in ss.top():
            self.assertLessEqual(h.guaranteed, truth[h.item], h)
            self.assertGreaterEqual(h.count, truth[h.item], h)
        n = sum(truth.values())
        for item, f in truth.items():
            if f > n / ss.capacity:
                self.assertIn(item, ss)

    def test_guarantees_on_zipf(self):
        stream = zipf_stream(30000, 3000, seed=1)
        ss = SpaceSaving(50)
        ss.update(stream)
        self.check_invariants(ss, collections.Counter(stream))

    def test_finds_true_top_items(self):
        stream = zipf_stream(30000, 3000, seed=2)
        truth = collections.Counter(stream)
        ss = SpaceSaving(100)
        ss.update(stream)
        want = [k for k, _ in truth.most_common(10)]
        got = [h.item for h in ss.top(10)]
        self.assertEqual(set(got), set(want))

    def test_exact_when_capacity_suffices(self):
        stream = zipf_stream(5000, 40, seed=3)
        ss = SpaceSaving(40)
        ss.update(stream)
        truth = collections.Counter(stream)
        for h in ss.top():
            self.assertEqual((h.count, h.error), (truth[h.item], 0))

    def test_heavy_hitters_complete(self):
        stream = zipf_stream(20000, 2000, seed=4)
        truth = collections.Counter(stream)
        ss = SpaceSaving(64)
        ss.update(stream)
        phi = 0.02
        reported = {h.item for h in ss.heavy_hitters(phi)}
        for item, f in truth.items():
            if f > phi * len(stream):
                self.assertIn(item, reported)
        with self.assertRaises(ValueError):
            ss.heavy_hitters(0)

    def test_weighted_and_adversarial_order(self):
        # Many distinct singletons interleaved with a few heavy items.
        ss = SpaceSaving(10)
        truth = collections.Counter()
        rng = random.Random(5)
        for i in range(5000):
            x = f"heavy{i % 3}" if rng.random() < 0.3 else f"noise{i}"
            w = rng.randint(1, 4)
            ss.add(x, w)
            truth[x] += w
        self.check_invariants(ss, truth)
        with self.assertRaises(ValueError):
            ss.add("x", 0)

    def test_merge_keeps_bounds(self):
        s1, s2 = zipf_stream(15000, 2000, seed=6), zipf_stream(15000, 2000, seed=7)
        a, b = SpaceSaving(80), SpaceSaving(80)
        a.update(s1)
        b.update(s2)
        a.merge(b)
        truth = collections.Counter(s1 + s2)
        self.assertEqual(a.total, 30000)
        for h in a.top():
            self.assertLessEqual(h.guaranteed, truth[h.item])
            self.assertGreaterEqual(h.count, truth[h.item])
        top5 = {k for k, _ in truth.most_common(5)}
        self.assertTrue(top5 <= {h.item for h in a.top(20)})

    def test_estimate_lookup(self):
        ss = SpaceSaving(5)
        ss.add("a", 3)
        self.assertEqual(ss.estimate("a").count, 3)
        self.assertIsNone(ss.estimate("zzz"))


if __name__ == "__main__":
    unittest.main()
