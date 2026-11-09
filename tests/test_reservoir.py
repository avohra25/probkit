import collections
import unittest

from probkit.reservoir import Reservoir, reservoir_sample


class ReservoirTest(unittest.TestCase):
    def test_short_stream_kept_whole(self):
        self.assertEqual(reservoir_sample(range(5), 10, seed=1), list(range(5)))

    def test_size_and_distinctness(self):
        s = reservoir_sample(range(100000), 50, seed=2)
        self.assertEqual(len(s), 50)
        self.assertEqual(len(set(s)), 50)

    def test_uniformity(self):
        # Each of 20 items should land in a size-5 sample ~25% of the time.
        hits = collections.Counter()
        trials = 6000
        for seed in range(trials):
            hits.update(reservoir_sample(range(20), 5, seed=seed))
        expected = trials * 5 / 20
        # Chi-square with 19 dof; 43.8 is the 99.9th percentile.
        chi2 = sum((hits[i] - expected) ** 2 / expected for i in range(20))
        self.assertLess(chi2, 43.8)

    def test_late_items_can_enter(self):
        late = sum(any(x >= 9000 for x in reservoir_sample(range(10000), 20, seed=s))
                   for s in range(200))
        # P(no item from the last 10%) = 0.9^20 ~ 12%, so most samples have one.
        self.assertGreater(late, 140)

    def test_seen_counter_and_validation(self):
        r = Reservoir(3, seed=0)
        r.update("abcdefg")
        self.assertEqual((r.seen, len(r)), (7, 3))
        with self.assertRaises(ValueError):
            Reservoir(0)


if __name__ == "__main__":
    unittest.main()
