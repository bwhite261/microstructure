"""Record Kraken v2 websocket market data (L2 book + trades) to hourly gzip files.

Each line in the output is `recv_timestamp<TAB>raw_message`. Raw messages are
kept verbatim so the order book can be reconstructed and checksum-validated
offline (see book.py).

Usage:
    python recorder.py --symbols BTC/USD ETH/USD --depth 10 --out data
"""

import argparse
import asyncio
import gzip
import json
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets

WS_URL = "wss://ws.kraken.com/v2"
RECORD_CHANNELS = {"instrument", "book", "trade"}

log = logging.getLogger("recorder")


class HourlyWriter:
    """Append (timestamp, raw json) lines to kraken-YYYYMMDD-HH.jsonl.gz, rotating hourly."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.hour = None
        self.file = None
        self.unflushed = 0

    def write(self, ts: float, raw: str) -> None:
        hour = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d-%H")
        if hour != self.hour:
            self.close()
            self.file = gzip.open(self.out_dir / f"kraken-{hour}.jsonl.gz", "at")
            self.hour = hour
        self.file.write(f"{ts:.6f}\t{raw}\n")
        self.unflushed += 1
        if self.unflushed >= 500:
            self.file.flush()
            self.unflushed = 0

    def close(self) -> None:
        if self.file:
            self.file.close()
            self.file = None
            self.hour = None


async def record(symbols: list[str], depth: int, writer: HourlyWriter) -> None:
    subscriptions = [
        {"method": "subscribe", "params": {"channel": "instrument"}},
        {"method": "subscribe", "params": {"channel": "book", "symbol": symbols, "depth": depth}},
        {"method": "subscribe", "params": {"channel": "trade", "symbol": symbols}},
    ]
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=20) as ws:
                for sub in subscriptions:
                    await ws.send(json.dumps(sub))
                log.info("connected: %s depth=%d", symbols, depth)
                backoff = 1.0
                count, last_report = 0, time.time()
                async for raw in ws:
                    ts = time.time()
                    msg = json.loads(raw)
                    if msg.get("method") == "subscribe" and not msg.get("success", True):
                        log.error("subscribe failed: %s", raw)
                    if msg.get("channel") in RECORD_CHANNELS:
                        writer.write(ts, raw)
                        count += 1
                    if ts - last_report >= 60:
                        log.info("recorded %d messages in last %.0fs", count, ts - last_report)
                        count, last_report = 0, ts
        except (websockets.ConnectionClosed, OSError) as e:
            log.warning("connection lost (%s), reconnecting in %.0fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["BTC/USD", "ETH/USD"])
    parser.add_argument("--depth", type=int, default=10, choices=[10, 25, 100, 500, 1000])
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    writer = HourlyWriter(args.out)
    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        asyncio.run(record(args.symbols, args.depth, writer))
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
