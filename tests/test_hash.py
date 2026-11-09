import random
import unittest

from probkit._hash import hash_pair, key_bytes, murmur3_128, murmur3_32

FOX = b"The quick brown fox jumps over the lazy dog"

# Reference values from Appleby's C++ implementation (cross-checked with mmh3).
VEC32 = [
    (b"", 0, 0x00000000),
    (b"", 1, 0x514E28B7),
    (b"", 0xFFFFFFFF, 0x81F16F39),
    (b"\xff\xff\xff\xff", 0, 0x76293B50),
    (b"!Ce\x87", 0, 0xF55B516B),
    (b"hello", 0, 0x248BFA47),
    (FOX, 0, 0x2E4FF723),
    (FOX, 42, 0x347CA102),
    (bytes(range(31)), 7, 0x157B85F4),
]
VEC128 = [
    (b"", 0, (0, 0)),
    (b"", 1, (0x4610ABE56EFF5CB5, 0x51622DAA78F83583)),
    (b"hello", 0, (0xCBD8A7B341BD9B02, 0x5B1E906A48AE1D19)),
    (FOX, 0, (0xE34BBC7BBC071B6C, 0x7A433CA9C49A9347)),
    (FOX, 42, (0x740DCF93FE0BD5D7, 0xC4546CF4EC705C8F)),
    (bytes(range(31)), 7, (0x04365954BE67F77E, 0x5A9E408D5359E11C)),
]


class MurmurTest(unittest.TestCase):
    def test_x86_32_vectors(self):
        for data, seed, want in VEC32:
            self.assertEqual(murmur3_32(data, seed), want, (data, seed))

    def test_x64_128_vectors(self):
        for data, seed, want in VEC128:
            self.assertEqual(murmur3_128(data, seed), want, (data, seed))

    def test_every_tail_length(self):
        # All tail lengths 0..15 must hash differently and deterministically.
        seen = set()
        for n in range(48):
            h = murmur3_128(bytes(range(n)))
            self.assertEqual(h, murmur3_128(bytes(range(n))))
            seen.add(h)
        self.assertEqual(len(seen), 48)

    def test_avalanche(self):
        # Flipping one input bit should flip ~half of the 128 output bits.
        rng = random.Random(3)
        total = trials = 0
        for _ in range(200):
            data = bytearray(rng.randbytes(20))
            a = murmur3_128(bytes(data))
            bit = rng.randrange(160)
            data[bit // 8] ^= 1 << (bit % 8)
            b = murmur3_128(bytes(data))
            total += bin(a[0] ^ b[0]).count("1") + bin(a[1] ^ b[1]).count("1")
            trials += 1
        self.assertAlmostEqual(total / trials, 64, delta=3)


class KeyTest(unittest.TestCase):
    def test_key_types(self):
        self.assertEqual(key_bytes("é"), "é".encode())
        self.assertEqual(key_bytes(b"x"), b"x")
        self.assertEqual(key_bytes(bytearray(b"x")), b"x")
        self.assertEqual(key_bytes(-12), b"-12")
        self.assertEqual(hash_pair("42"), hash_pair(42))

    def test_rejects_other_types(self):
        for bad in (1.5, None, True, ("a",)):
            with self.assertRaises(TypeError):
                key_bytes(bad)


if __name__ == "__main__":
    unittest.main()
