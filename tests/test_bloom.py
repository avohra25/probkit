import math
import unittest

from probkit.bloom import BloomFilter, CountingBloomFilter, optimal_params


class ParamsTest(unittest.TestCase):
    def test_textbook_values(self):
        # n=1000, p=1% -> m ~ 9586 bits, k = 7
        m, k = optimal_params(1000, 0.01)
        self.assertEqual((m, k), (9586, 7))

    def test_validation(self):
        for cap, p in ((0, 0.1), (10, 0), (10, 1), (10, 1.5)):
            with self.assertRaises(ValueError):
                optimal_params(cap, p)


class BloomTest(unittest.TestCase):
    N = 5000

    def build(self, fpr=0.01, seed=0):
        bf = BloomFilter(capacity=self.N, fpr=fpr, seed=seed)
        bf.update(f"member-{i}" for i in range(self.N))
        return bf

    def test_no_false_negatives(self):
        bf = self.build()
        self.assertTrue(all(f"member-{i}" in bf for i in range(self.N)))

    def test_false_positive_rate_near_target(self):
        for fpr in (0.05, 0.01):
            bf = self.build(fpr)
            trials = 20000
            fp = sum(f"other-{i}" in bf for i in range(trials))
            observed = fp / trials
            self.assertLess(observed, fpr * 1.5, fpr)
            self.assertGreater(observed, fpr * 0.5, fpr)
            self.assertAlmostEqual(bf.estimated_fpr(), fpr, delta=fpr * 0.3)

    def test_add_reports_novelty(self):
        bf = BloomFilter(100, 0.01)
        self.assertTrue(bf.add("x"))
        self.assertFalse(bf.add("x"))

    def test_estimate_count(self):
        bf = self.build()
        self.assertAlmostEqual(bf.estimate_count(), self.N, delta=self.N * 0.03)
        self.assertEqual(len(BloomFilter(10, 0.1)), 0)

    def test_union_equals_inserting_both(self):
        a = BloomFilter(2000, 0.01)
        b = BloomFilter(2000, 0.01)
        both = BloomFilter(2000, 0.01)
        for i in range(800):
            (a if i % 2 else b).add(i)
            both.add(i)
        self.assertEqual(a | b, both)

    def test_intersection_contains_common_keys(self):
        a = BloomFilter(2000, 0.01)
        b = BloomFilter(2000, 0.01)
        a.update(range(0, 600))
        b.update(range(300, 900))
        inter = a & b
        self.assertTrue(all(i in inter for i in range(300, 600)))
        outside = sum(i in inter for i in range(10000, 20000))
        self.assertLess(outside, 300)

    def test_incompatible_combination(self):
        with self.assertRaises(ValueError):
            BloomFilter(100, 0.01) | BloomFilter(200, 0.01)
        with self.assertRaises(ValueError):
            BloomFilter(100, 0.01, seed=1) | BloomFilter(100, 0.01, seed=2)
        with self.assertRaises(TypeError):
            BloomFilter(100, 0.01) | set()

    def test_seed_changes_layout(self):
        self.assertNotEqual(self.build(seed=1), self.build(seed=2))

    def test_round_trip(self):
        bf = self.build()
        back = BloomFilter.from_bytes(bf.to_bytes())
        self.assertEqual(back, bf)
        self.assertEqual(back.inserts, self.N)
        self.assertIn("member-17", back)

    def test_corrupt_input(self):
        data = self.build().to_bytes()
        with self.assertRaises(ValueError):
            BloomFilter.from_bytes(data[:10])
        with self.assertRaises(ValueError):
            BloomFilter.from_bytes(b"XXXX" + data[4:])
        with self.assertRaises(ValueError):
            BloomFilter.from_bytes(data[:-1])


class CountingBloomTest(unittest.TestCase):
    def test_add_remove_cycle(self):
        cbf = CountingBloomFilter(1000, 0.01)
        for i in range(1000):
            cbf.add(i)
        for i in range(0, 1000, 2):
            cbf.remove(i)
        self.assertEqual(len(cbf), 500)
        self.assertTrue(all(i in cbf for i in range(1, 1000, 2)))
        gone = sum(i in cbf for i in range(0, 1000, 2))
        self.assertLess(gone, 30)  # only false positives remain

    def test_remove_absent_raises(self):
        cbf = CountingBloomFilter(100, 0.01)
        with self.assertRaises(KeyError):
            cbf.remove("ghost")
        self.assertFalse(cbf.discard("ghost"))

    def test_count_is_upper_bound(self):
        cbf = CountingBloomFilter(500, 0.01)
        for i in range(200):
            for _ in range(i % 5 + 1):
                cbf.add(i)
        for i in range(200):
            self.assertGreaterEqual(cbf.count(i), i % 5 + 1)

    def test_saturation_never_causes_false_negative(self):
        cbf = CountingBloomFilter(10, 0.1)
        for _ in range(300):
            cbf.add("hot")
        cbf.add("cold")
        for _ in range(300):
            cbf.discard("hot")
        self.assertIn("cold", cbf)


if __name__ == "__main__":
    unittest.main()
