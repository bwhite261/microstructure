from decimal import Decimal
from pathlib import Path

import pytest

from book import Book, ChecksumMismatch, _fmt, replay

FIXTURE = Path(__file__).parent / "data" / "fixture.jsonl.gz"


def make_book():
    book = Book("TEST/USD", depth=3, price_precision=1, qty_precision=8)
    book.apply(
        {
            "bids": [{"price": Decimal("100.0"), "qty": Decimal("1")},
                     {"price": Decimal("99.9"), "qty": Decimal("2")},
                     {"price": Decimal("99.8"), "qty": Decimal("3")}],
            "asks": [{"price": Decimal("100.1"), "qty": Decimal("1")},
                     {"price": Decimal("100.2"), "qty": Decimal("2")},
                     {"price": Decimal("100.3"), "qty": Decimal("3")}],
        },
        snapshot=True,
    )
    return book


def test_update_and_remove_levels():
    book = make_book()
    book.apply({"bids": [{"price": Decimal("100.0"), "qty": Decimal("5")}]}, snapshot=False)
    assert book.best_bid() == (Decimal("100.0"), Decimal("5"))
    book.apply({"bids": [{"price": Decimal("100.0"), "qty": Decimal("0")}]}, snapshot=False)
    assert book.best_bid() == (Decimal("99.9"), Decimal("2"))


def test_new_level_pushes_out_worst():
    book = make_book()
    book.apply({"asks": [{"price": Decimal("100.05"), "qty": Decimal("1")}]}, snapshot=False)
    assert book.best_ask()[0] == Decimal("100.05")
    assert len(book.asks) == 3
    assert Decimal("100.3") not in book.asks


def test_depth_qty():
    book = make_book()
    assert book.depth_qty("bids", 2) == Decimal("3")
    assert book.depth_qty("asks", 3) == Decimal("6")


def test_checksum_format():
    assert _fmt(Decimal("64122.4"), 1) == "641224"
    assert _fmt(Decimal("0.00311748"), 8) == "311748"
    assert _fmt(Decimal("1819.08"), 2) == "181908"


def test_replay_validates_live_fixture():
    """Every book message in the recorded fixture must reproduce Kraken's checksum."""
    events = list(replay([FIXTURE], validate=True))
    assert sum(1 for _, kind, _ in events if kind == "book") > 1000


def test_checksum_detects_corruption():
    """A one-satoshi error in any level must change the checksum."""
    events = replay([FIXTURE], validate=True)
    book = next(payload for _, kind, payload in events if kind == "book")
    good = book.checksum()
    book.bids[min(book.bids)] += Decimal("0.00000001")
    assert book.checksum() != good
