"""Command-line interface: ``probkit <command> ...`` (or ``python -m probkit``)."""

from __future__ import annotations

import argparse
import sys
from typing import Iterator

from . import __version__
from ._hash import murmur3_128, murmur3_32
from .bloom import BloomFilter
from .hll import HyperLogLog
from .reservoir import reservoir_sample
from .topk import SpaceSaving


def _lines(path: str) -> Iterator[str]:
    stream = sys.stdin if path == "-" else open(path, encoding="utf-8", errors="surrogateescape")
    try:
        for line in stream:
            line = line.rstrip("\r\n")
            if line:
                yield line
    finally:
        if stream is not sys.stdin:
            stream.close()


def _key(line: str) -> bytes:
    return line.encode("utf-8", "surrogateescape")


def cmd_hash(a) -> int:
    data = a.text.encode("utf-8")
    h1, h2 = murmur3_128(data, a.seed)
    print(f"murmur3_32   {murmur3_32(data, a.seed):08x}")
    print(f"murmur3_128  {h1:016x}{h2:016x}")
    return 0


def cmd_count(a) -> int:
    h = HyperLogLog(a.precision)
    n = 0
    for line in _lines(a.file):
        h.add(_key(line))
        n += 1
    est = h.count()
    print(f"lines        {n}")
    print(f"distinct     ~{est:.0f}  (±{100 * h.relative_error:.1f}% at 1σ, "
          f"{h.m} registers)")
    return 0


def cmd_top(a) -> int:
    ss = SpaceSaving(max(a.capacity, a.k))
    for line in _lines(a.file):
        ss.add(line)
    rows = ss.top(a.k)
    if not rows:
        return 0
    width = max(len(str(h.count)) for h in rows)
    for h in rows:
        bound = "" if h.error == 0 else f"  (≥{h.guaranteed})"
        print(f"{h.count:>{width}}  {h.item}{bound}")
    return 0


def cmd_sample(a) -> int:
    for line in reservoir_sample(_lines(a.file), a.k, seed=a.seed):
        print(line)
    return 0


def cmd_bloom_build(a) -> int:
    lines = list(_lines(a.file))
    capacity = a.capacity or max(1, len(set(lines)))
    bf = BloomFilter(capacity, a.fpr)
    for line in lines:
        bf.add(_key(line))
    data = bf.to_bytes()
    with open(a.output, "wb") as fh:
        fh.write(data)
    print(f"wrote {a.output}: {len(lines)} keys, {bf.num_bits} bits, "
          f"{bf.num_hashes} hashes, {len(data)} bytes, est. fpr {bf.estimated_fpr():.4%}",
          file=sys.stderr)
    return 0


def cmd_bloom_check(a) -> int:
    with open(a.filter, "rb") as fh:
        bf = BloomFilter.from_bytes(fh.read())
    items = a.items or list(_lines("-"))
    missing = 0
    for item in items:
        hit = _key(item) in bf
        missing += not hit
        print(f"{'maybe' if hit else 'no':5}  {item}")
    return 1 if a.strict and missing else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="probkit", description="Probabilistic data structure tools.")
    p.add_argument("--version", action="version", version=f"probkit {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("hash", help="print MurmurHash3 digests of a string")
    s.add_argument("text")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_hash)

    s = sub.add_parser("count", help="estimate distinct lines with HyperLogLog")
    s.add_argument("file", nargs="?", default="-")
    s.add_argument("-p", "--precision", type=int, default=14)
    s.set_defaults(func=cmd_count)

    s = sub.add_parser("top", help="most frequent lines via Space-Saving")
    s.add_argument("file", nargs="?", default="-")
    s.add_argument("-k", type=int, default=10)
    s.add_argument("-c", "--capacity", type=int, default=1000)
    s.set_defaults(func=cmd_top)

    s = sub.add_parser("sample", help="uniform random sample of lines")
    s.add_argument("file", nargs="?", default="-")
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--seed", type=int)
    s.set_defaults(func=cmd_sample)

    b = sub.add_parser("bloom", help="build or query a Bloom filter file")
    bsub = b.add_subparsers(dest="bloom_command", required=True)
    s = bsub.add_parser("build", help="build a filter from lines")
    s.add_argument("file", nargs="?", default="-")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--fpr", type=float, default=0.01)
    s.add_argument("--capacity", type=int, help="defaults to the number of distinct lines")
    s.set_defaults(func=cmd_bloom_build)
    s = bsub.add_parser("check", help="test items (or stdin lines) against a filter")
    s.add_argument("filter")
    s.add_argument("items", nargs="*")
    s.add_argument("--strict", action="store_true", help="exit 1 if any item is absent")
    s.set_defaults(func=cmd_bloom_check)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:  # pragma: no cover - e.g. `probkit top | head`
        return 0
    except (OSError, ValueError) as exc:
        print(f"probkit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
