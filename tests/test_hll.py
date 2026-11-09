import unittest

from probkit.hll import HyperLogLog


class HLLTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(HyperLogLog(10).count(), 0.0)

    def test_small_cardinalities_are_near_exact(self):
        # Linear counting keeps tiny sets almost exact.
        for n in (1, 2, 10, 100):
            h = HyperLogLog(12)
            h.update(range(n))
            self.assertAlmostEqual(h.count(), n, delta=max(1, n * 0.02))

    def test_error_within_three_sigma(self):
        for p in (10, 12, 14):
            for n in (1000, 30000):
                h = HyperLogLog(p)
                h.update(f"user:{i}" for i in range(n))
                rel = abs(h.count() - n) / n
                self.assertLess(rel, 3 * h.relative_error, (p, n, h.count()))

    def test_mean_error_is_unbiased(self):
        # Averaged over seeds the estimate should centre on the truth.
        n, errs = 20000, []
        for seed in range(12):
            h = HyperLogLog(10, seed=seed)
            h.update(range(n))
            errs.append((h.count() - n) / n)
        self.assertLess(abs(sum(errs) / len(errs)), 0.02)

    def test_duplicates_do_not_count(self):
        h = HyperLogLog(12)
        for _ in range(20):
            h.update(range(500))
        self.assertAlmostEqual(h.count(), 500, delta=25)
        self.assertFalse(h.add(7))

    def test_merge_equals_combined_stream(self):
        a, b, both = HyperLogLog(12), HyperLogLog(12), HyperLogLog(12)
        a.update(range(0, 6000))
        b.update(range(4000, 10000))
        both.update(range(10000))
        self.assertEqual(a | b, both)
        a.merge(b)
        self.assertEqual(a, both)

    def test_intersection_and_jaccard(self):
        a, b = HyperLogLog(14), HyperLogLog(14)
        a.update(range(0, 20000))
        b.update(range(10000, 30000))
        self.assertAlmostEqual(a.intersection_estimate(b), 10000, delta=1500)
        self.assertAlmostEqual(a.jaccard_estimate(b), 1 / 3, delta=0.05)

    def test_incompatible_merge(self):
        with self.assertRaises(ValueError):
            HyperLogLog(10).merge(HyperLogLog(11))
        with self.assertRaises(ValueError):
            HyperLogLog(10, seed=1).merge(HyperLogLog(10))
        with self.assertRaises(TypeError):
            HyperLogLog(10).merge(set())

    def test_round_trip_and_validation(self):
        h = HyperLogLog(8, seed=5)
        h.update(range(3000))
        back = HyperLogLog.from_bytes(h.to_bytes())
        self.assertEqual(back, h)
        self.assertEqual(back.count(), h.count())
        data = h.to_bytes()
        for bad in (data[:3], b"XXXX" + data[4:], data[:-1], data[:-1] + b"\xff"):
            with self.assertRaises(ValueError):
                HyperLogLog.from_bytes(bad)
        for p in (3, 19):
            with self.assertRaises(ValueError):
                HyperLogLog(p)

    def test_rho_extremes(self):
        h = HyperLogLog(4)
        h._add_hash(0)  # all-zero tail: maximum rank
        self.assertEqual(h._reg[0], 64 - 4 + 1)
        h._add_hash((0b1111 << 60) | (1 << 59))
        self.assertEqual(h._reg[15], 1)


if __name__ == "__main__":
    unittest.main()
