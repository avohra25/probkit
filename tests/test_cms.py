import collections
import random
import unittest

from probkit.cms import CountMinSketch


def zipf_stream(n, universe, s=1.1, seed=0):
    rng = random.Random(seed)
    weights = [1 / (r ** s) for r in range(1, universe + 1)]
    return rng.choices(range(universe), weights=weights, k=n)


class CMSTest(unittest.TestCase):
    def test_dimensions(self):
        cms = CountMinSketch(0.01, 0.01)
        self.assertEqual((cms.width, cms.depth), (272, 5))
        self.assertLessEqual(cms.epsilon, 0.01)
        self.assertLessEqual(cms.delta, 0.01)

    def check_bounds(self, conservative):
        stream = zipf_stream(30000, 5000, seed=1)
        truth = collections.Counter(stream)
        cms = CountMinSketch(0.002, 0.01, conservative=conservative)
        for x in stream:
            cms.add(x)
        self.assertEqual(cms.total, len(stream))
        bound = cms.epsilon * cms.total
        over = []
        for x in range(5000):
            est = cms[x]
            self.assertGreaterEqual(est, truth[x])  # never underestimates
            over.append(est - truth[x])
        violations = sum(o > bound for o in over)
        self.assertLessEqual(violations / len(over), cms.delta * 2)
        return sum(over)

    def test_standard_update_bounds(self):
        self.check_bounds(False)

    def test_conservative_update_is_tighter(self):
        self.assertLess(self.check_bounds(True), self.check_bounds(False))

    def test_weighted_add(self):
        cms = CountMinSketch(0.01, 0.01)
        self.assertEqual(cms.add("a", 5), 5)
        cms.add("a", 0)
        self.assertEqual(cms["a"], 5)
        with self.assertRaises(ValueError):
            cms.add("a", -1)

    def test_merge(self):
        a, b, both = (CountMinSketch(0.01, 0.01) for _ in range(3))
        for i, x in enumerate(zipf_stream(4000, 300, seed=2)):
            (a if i % 3 else b).add(x)
            both.add(x)
        a.merge(b)
        self.assertEqual(a, both)
        with self.assertRaises(ValueError):
            a.merge(CountMinSketch(0.1, 0.01))

    def test_inner_product_upper_bound(self):
        s1, s2 = zipf_stream(5000, 400, seed=3), zipf_stream(5000, 400, seed=4)
        a, b = CountMinSketch(0.005, 0.01), CountMinSketch(0.005, 0.01)
        for x in s1:
            a.add(x)
        for x in s2:
            b.add(x)
        c1, c2 = collections.Counter(s1), collections.Counter(s2)
        exact = sum(c1[k] * c2[k] for k in c1)
        est = a.inner_product(b)
        self.assertGreaterEqual(est, exact)
        self.assertLessEqual(est - exact, a.epsilon * a.total * b.total)

    def test_round_trip(self):
        cms = CountMinSketch(0.01, 0.05, conservative=True, seed=11)
        for x in zipf_stream(2000, 100):
            cms.add(x)
        back = CountMinSketch.from_bytes(cms.to_bytes())
        self.assertEqual(back, cms)
        self.assertTrue(back.conservative)
        data = cms.to_bytes()
        for bad in (data[:4], b"NOPE" + data[4:], data[:-8]):
            with self.assertRaises(ValueError):
                CountMinSketch.from_bytes(bad)


if __name__ == "__main__":
    unittest.main()
