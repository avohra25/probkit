import contextlib
import io
import os
import tempfile
import unittest

from probkit.cli import main


def run(*argv, stdin=None):
    out, err = io.StringIO(), io.StringIO()
    import sys
    old = sys.stdin
    if stdin is not None:
        sys.stdin = io.StringIO(stdin)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
    finally:
        sys.stdin = old
    return code, out.getvalue(), err.getvalue()


class CLITest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.words = os.path.join(self.tmp.name, "words.txt")
        lines = [f"w{i % 700}" for i in range(5000)] + ["hot"] * 400
        with open(self.words, "w") as fh:
            fh.write("\n".join(lines) + "\n")

    def test_hash(self):
        code, out, _ = run("hash", "hello")
        self.assertEqual(code, 0)
        self.assertIn("248bfa47", out)
        self.assertIn("cbd8a7b341bd9b025b1e906a48ae1d19", out)

    def test_count(self):
        code, out, _ = run("count", self.words, "-p", "12")
        self.assertEqual(code, 0)
        est = int(out.split("~")[1].split()[0])
        self.assertAlmostEqual(est, 701, delta=30)

    def test_top(self):
        code, out, _ = run("top", self.words, "-k", "1")
        self.assertEqual(code, 0)
        self.assertEqual(out.split(), ["400", "hot"])

    def test_sample_stdin(self):
        code, out, _ = run("sample", "-k", "3", "--seed", "1", stdin="a\nb\nc\nd\ne\n")
        self.assertEqual(code, 0)
        self.assertEqual(len(out.split()), 3)

    def test_bloom_build_and_check(self):
        flt = os.path.join(self.tmp.name, "w.bloom")
        code, _, err = run("bloom", "build", self.words, "-o", flt, "--fpr", "0.001")
        self.assertEqual(code, 0)
        self.assertIn("5400 keys", err)
        code, out, _ = run("bloom", "check", flt, "w5", "hot", "definitely-not-there")
        self.assertEqual(code, 0)
        self.assertEqual([l.split()[0] for l in out.splitlines()], ["maybe", "maybe", "no"])
        code, _, _ = run("bloom", "check", flt, "--strict", "nope")
        self.assertEqual(code, 1)
        code, out, _ = run("bloom", "check", flt, stdin="w1\nnah\n")
        self.assertEqual(out.split(), ["maybe", "w1", "no", "nah"])

    def test_errors(self):
        code, _, err = run("count", os.path.join(self.tmp.name, "missing.txt"))
        self.assertEqual(code, 2)
        self.assertIn("probkit:", err)
        bogus = os.path.join(self.tmp.name, "bogus.bloom")
        with open(bogus, "wb") as fh:
            fh.write(b"not a filter")
        code, _, err = run("bloom", "check", bogus, "x")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
