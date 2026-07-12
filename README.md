# microstructure

Do order-book imbalance and order-flow features predict short-horizon crypto
price moves — and does the signal survive trading costs?

An end-to-end pipeline against live Kraken market data:

1. **`recorder.py`** streams the L2 book (top 10 levels) and trades for
   BTC/USD and ETH/USD over websocket and archives every raw message.
2. **`book.py`** reconstructs the order book from snapshots + deltas and
   verifies it against the CRC32 checksum Kraken embeds in every book message
   — every bookkeeping bug shows up immediately as a checksum mismatch.
3. **`features.py`** aggregates events into 1-second bars: order-flow
   imbalance, book imbalance at depths 1/5/10, spread, signed trade flow,
   realized volatility.
4. **`models.py`** runs logistic regression and gradient-boosted trees with
   chronological walk-forward validation on direction labels at 1s/5s/30s.
5. **`backtest.py`** prices every signal three ways: mid-to-mid (is there
   signal?), crossing the spread (can it trade?), and spread + taker fees
   (can a retail taker profit?).

## Correctness

The book engine replays the recorded feed and recomputes Kraken's checksum
after **every** message — a full session validates with zero mismatches at
~50k events/sec in pure Python. The checksum covers price *and* quantity of
all ten levels per side at exact decimal precision, so it catches misapplied
deltas, truncation errors, and float drift (prices are handled as `Decimal`).

## Method notes

- **No lookahead.** Bar *t* contains only events with timestamps in
  [t, t+1); its features use the book state as of the last event in the bar.
  Labels are forward log mid returns measured from bar close to bar close.
- **Chronological validation with an embargo.** `TimeSeriesSplit` trains on
  the past and tests on the future; a 30-bar `gap` is purged between the two
  so the model is never scored on rows adjacent to its training data.
  Shuffled CV on an autocorrelated series leaks both ways.
- **Direction, not magnitude.** The label is the sign of the forward mid
  move, flat bars dropped; AUC is compared against the 0.5 no-skill baseline.
- **Attribution guards against a momentum artifact.** `models.py` reports AUC
  for feature subsets: if the last return alone predicted direction, the
  "signal" would just be autocorrelation. It doesn't — see below.

## Results

Numbers below are from a ~5.5-hour session (≈30k one-second bars per symbol);
everything regenerates with `python models.py && python backtest.py` as the
recorder accumulates more data.

**Direction prediction** (5s horizon, embargoed walk-forward, logistic):

| symbol | AUC | book features only | imb1 alone | last return alone |
|---|---|---|---|---|
| BTC/USD | 0.93 | 0.94 | 0.95 | 0.51 |
| ETH/USD | 0.64 | 0.64 | 0.64 | 0.53 |

Two things to read here. First, the signal is **order-book imbalance**, not
momentum: `imb1` alone reproduces nearly the full AUC while the last return
alone is barely above 0.5. Second, BTC scores far higher than ETH because of
**tick size** — BTC/USD moves in $0.10 ticks on a ~$64k price, so the mid only
changes when a best level clears, and best-level imbalance mechanically
telegraphs which side is about to clear. ETH's finer relative tick makes the
relationship noisier and the AUC closer to the values reported in the
microstructure literature.

**But is it tradeable?** The three-way backtest (threshold 0.6, ~600 trades):

| scenario | BTC avg/trade | ETH avg/trade |
|---|---|---|
| mid-to-mid (no cost) | +0.09 bps | +0.13 bps |
| crossing the spread | +0.07 bps | −0.05 bps |
| + 26 bps taker fee/side | −51.9 bps | −52.0 bps |

The high AUC and the tiny per-trade edge are consistent, and reconciling them
is the whole point: imbalance predicts *direction* well, but the predicted
move is a fraction of a basis point (often a single tick, and most 5s windows
don't move at all). A retail taker pays ~52 bps round trip to capture ~0.1 bp.
**The signal is real and nearly mechanical; the profit is not.** Short-horizon
edge of this kind is only harvestable by market makers who *earn* the spread
instead of paying it — which is exactly why latency and queue position matter
in real markets.

## Run it

```bash
pip install -r requirements.txt
python recorder.py                    # leave running; ~sub-GB per day
python book.py                        # replay + validate every checksum
python features.py                    # build 1s bars -> data/features-*.csv
python models.py                      # embargoed walk-forward + feature attribution
python backtest.py --horizon 5        # cost-aware PnL + equity curves
pytest                                # book engine + feature unit tests
```

`tests/data/fixture.jsonl.gz` is a slice of a real recording, so the test
suite validates the engine against genuine exchange checksums.

## What I'd do next

- Maker-side simulation with queue-position modeling (earn the spread
  instead of paying it) — requires L3 or fill data to be honest.
- Deeper features: multi-level OFI, trade-size distributions, cross-asset
  lead-lag (BTC leads ETH at these horizons).
- A C++ book engine for line-rate throughput.

## License

MIT
