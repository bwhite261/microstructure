"""L2 order book reconstruction from recorded Kraken v2 feeds.

Books are rebuilt by applying deltas to depth snapshots and verified against
the CRC32 checksum Kraken includes in every book message, so any bookkeeping
error is caught immediately. Prices/quantities are parsed as Decimal because
the checksum is defined over exact decimal strings.

Run directly to validate a recording:
    python book.py data/*.jsonl.gz
"""

import gzip
import json
import sys
import time
import zlib
from decimal import Decimal
from pathlib import Path


class ChecksumMismatch(Exception):
    pass


def _fmt(value: Decimal, precision: int) -> str:
    """Kraken checksum encoding: fixed precision, no decimal point, no leading zeros."""
    return f"{value:.{precision}f}".replace(".", "").lstrip("0")


class Book:
    """One symbol's depth-limited L2 book: price -> aggregated quantity per side."""

    def __init__(self, symbol: str, depth: int, price_precision: int, qty_precision: int):
        self.symbol = symbol
        self.depth = depth
        self.price_precision = price_precision
        self.qty_precision = qty_precision
        self.bids: dict[Decimal, Decimal] = {}
        self.asks: dict[Decimal, Decimal] = {}

    def apply(self, entry: dict, snapshot: bool) -> None:
        if snapshot:
            self.bids = {level["price"]: level["qty"] for level in entry["bids"]}
            self.asks = {level["price"]: level["qty"] for level in entry["asks"]}
            return
        for side_name, side in (("bids", self.bids), ("asks", self.asks)):
            for level in entry.get(side_name, ()):
                if level["qty"] == 0:
                    side.pop(level["price"], None)
                else:
                    side[level["price"]] = level["qty"]
        # a new level inside the top N pushes the old Nth out without a delete
        for price in sorted(self.bids, reverse=True)[self.depth :]:
            del self.bids[price]
        for price in sorted(self.asks)[self.depth :]:
            del self.asks[price]

    def checksum(self) -> int:
        parts = [
            _fmt(price, self.price_precision) + _fmt(self.asks[price], self.qty_precision)
            for price in sorted(self.asks)[:10]
        ] + [
            _fmt(price, self.price_precision) + _fmt(self.bids[price], self.qty_precision)
            for price in sorted(self.bids, reverse=True)[:10]
        ]
        return zlib.crc32("".join(parts).encode())

    def best_bid(self) -> tuple[Decimal, Decimal]:
        price = max(self.bids)
        return price, self.bids[price]

    def best_ask(self) -> tuple[Decimal, Decimal]:
        price = min(self.asks)
        return price, self.asks[price]

    def depth_qty(self, side: str, n: int) -> Decimal:
        """Total quantity on the best n levels of 'bids' or 'asks'."""
        levels = self.bids if side == "bids" else self.asks
        prices = sorted(levels, reverse=(side == "bids"))[:n]
        return sum((levels[p] for p in prices), Decimal(0))


def _lines(path: Path):
    """Yield lines from a recording, stopping cleanly if it is still being written."""
    with gzip.open(path, "rt") as f:
        while True:
            try:
                line = f.readline()
            except EOFError:
                return
            if not line:
                return
            yield line


def replay(paths: list[Path], validate: bool = True):
    """Replay recordings in order, yielding (ts, kind, payload) events.

    kind is 'book' (payload = the updated Book) or 'trade' (payload = trade dict).
    Paths must start at a recording session boundary so the instrument snapshot
    and book snapshots precede any updates.
    """
    precisions: dict[str, tuple[int, int]] = {}
    books: dict[str, Book] = {}
    for path in paths:
        for line in _lines(path):
            try:
                ts, raw = line.split("\t", 1)
                msg = json.loads(raw, parse_float=Decimal)
            except ValueError:  # partial final line of a live file
                break
            channel = msg.get("channel")
            if channel == "instrument":
                for pair in msg["data"]["pairs"]:
                    precisions[pair["symbol"]] = (pair["price_precision"], pair["qty_precision"])
            elif channel == "book":
                for entry in msg["data"]:
                    symbol = entry["symbol"]
                    if msg["type"] == "snapshot":
                        if symbol not in precisions:
                            raise ValueError(f"no instrument data before {symbol} snapshot; "
                                             "replay from the start of the session")
                        books[symbol] = Book(symbol, len(entry["bids"]), *precisions[symbol])
                    book = books[symbol]
                    book.apply(entry, msg["type"] == "snapshot")
                    if validate and book.checksum() != entry["checksum"]:
                        raise ChecksumMismatch(f"{symbol} at ts {ts}")
                    yield float(ts), "book", book
            elif channel == "trade":
                for trade in msg["data"]:
                    yield float(ts), "trade", trade


def main() -> None:
    paths = sorted(Path(p) for p in sys.argv[1:]) or sorted(Path("data").glob("*.jsonl.gz"))
    counts = {"book": 0, "trade": 0}
    t0 = time.time()
    for _, kind, _ in replay(paths):
        counts[kind] += 1
    elapsed = time.time() - t0
    total = sum(counts.values())
    print(f"validated {counts['book']:,} book updates (every checksum passed) and "
          f"{counts['trade']:,} trades in {elapsed:.1f}s ({total / elapsed:,.0f} events/s)")


if __name__ == "__main__":
    main()
