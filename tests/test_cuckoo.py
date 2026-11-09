import random
import unittest

from probkit.cuckoo import CuckooFilter, CuckooFilterFull


class CuckooTest(unittest.TestCase):
    def test_sizing(self):
        cf = CuckooFilter(1000)
        self.assertEqual(cf.num_buckets & (cf.num_buckets - 1), 0)
        self.assertGreaterEqual(cf.capacity * 0.95, 1000)

    def test_alt_index_is_involution(self):
        cf = CuckooFilter(4096)
        rng = random.Random(1)
        for _ in range(2000):
            i = rng.randrange(cf.num_buckets)
            fp = rng.randrange(1, 1 << cf.fingerprint_bits)
            self.assertEqual(cf._alt(cf._alt(i, fp), fp), i)

    def test_no_false_negatives_at_high_load(self):
        cf = CuckooFilter(5000)
        for i in range(5000):
            cf.add(f"k{i}")
        self.assertGreater(cf.load_factor(), 0.6)
        self.assertTrue(all(f"k{i}" in cf for i in range(5000)))

    def test_false_positive_rate(self):
        cf = CuckooFilter(5000, fingerprint_bits=10)
        for i in range(5000):
            cf.add(f"k{i}")
        trials = 20000
        fp = sum(f"x{i}" in cf for i in range(trials)) / trials
        self.assertLess(fp, cf.fpr_bound())
        self.assertGreater(fp, 0)

    def test_delete(self):
        cf = CuckooFilter(2000)
        for i in range(2000):
            cf.add(i)
        for i in range(0, 2000, 2):
            cf.remove(i)
        self.assertEqual(len(cf), 1000)
        self.assertTrue(all(i in cf for i in range(1, 2000, 2)))
        remaining = sum(i in cf for i in range(0, 2000, 2))
        self.assertLess(remaining, 20)

    def test_duplicates_counted(self):
        cf = CuckooFilter(100)
        cf.add("dup")
        cf.add("dup")
        cf.remove("dup")
        self.assertIn("dup", cf)
        cf.remove("dup")
        self.assertNotIn("dup", cf)
        with self.assertRaises(KeyError):
            cf.remove("dup")
        self.assertFalse(cf.discard("dup"))

    def test_full_filter_never_loses_keys(self):
        cf = CuckooFilter(64, max_kicks=50)
        added = []
        with self.assertRaises(CuckooFilterFull):
            for i in range(10000):
                cf.add(i)
                added.append(i)
        # Every key that was acknowledged, plus the one that triggered the
        # failure (parked as the victim or displaced into it), is still found.
        self.assertTrue(all(k in cf for k in added))
        self.assertIn(len(added), range(int(cf.capacity * 0.8), cf.capacity + 1))
        self.assertEqual(len(cf), len(added) + 1)
        self.assertFalse(cf.try_add("one-more"))
        # Deleting frees room, and the parked victim moves back into the table.
        cf.remove(added[0])
        self.assertIsNone(cf._victim)
        self.assertTrue(all(k in cf for k in added[1:]))

    def test_round_trip(self):
        cf = CuckooFilter(500, seed=9)
        for i in range(400):
            cf.add(f"w{i}")
        back = CuckooFilter.from_bytes(cf.to_bytes())
        self.assertEqual(back, cf)
        self.assertEqual(len(back), 400)
        self.assertTrue(all(f"w{i}" in back for i in range(400)))
        back.remove("w3")
        self.assertNotIn("w3", back)

    def test_corrupt_input(self):
        data = CuckooFilter(100).to_bytes()
        for bad in (data[:5], b"NOPE" + data[4:], data[:-4]):
            with self.assertRaises(ValueError):
                CuckooFilter.from_bytes(bad)

    def test_validation(self):
        with self.assertRaises(ValueError):
            CuckooFilter(0)
        with self.assertRaises(ValueError):
            CuckooFilter(10, fingerprint_bits=2)


if __name__ == "__main__":
    unittest.main()
