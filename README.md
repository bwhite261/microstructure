# microstructure

Do order-book imbalance and order-flow features predict short-horizon crypto
price moves — and does the signal survive trading costs?

An end-to-end pipeline against live Kraken market data:

1. **`recorder.py`** streams the L2 book (top 10 levels) and trades for
   BTC, ETH, SOL, and XRP over websocket and archives every raw message.
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
6. **`leadlag.py`** measures cross-asset lead-lag between two assets and tests
   whether one's features improve a direction model for the other.

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

Numbers below are from a ~34-hour, multi-session recording (~100k one-second
bars each for BTC and ETH); everything regenerates as the recorder accumulates
more data. SOL and XRP were added later and are still building history.

**Direction prediction** (5s horizon, embargoed walk-forward, logistic):

| symbol | AUC | book features only | imb1 alone | last return alone |
|---|---|---|---|---|
| BTC/USD | 0.90 | 0.90 | 0.89 | 0.55 |
| ETH/USD | 0.59 | 0.59 | 0.58 | 0.53 |

Two things to read here. First, the signal is **order-book imbalance**, not
momentum: `imb1` alone reproduces nearly the full AUC while the last return
alone is barely above 0.5. Second, BTC scores far higher than ETH because of
**tick size** — BTC/USD moves in $0.10 ticks on a ~$64k price, so the mid only
changes when a best level clears, and best-level imbalance mechanically
telegraphs which side is about to clear. ETH's finer relative tick makes the
relationship noisier and the AUC closer to the values reported in the
microstructure literature.

**But is it tradeable?** The three-way backtest (threshold 0.6, ~3–4k trades):

| scenario | BTC avg/trade | ETH avg/trade |
|---|---|---|
| mid-to-mid (no cost) | +0.30 bps | +0.41 bps |
| crossing the spread | +0.27 bps | +0.27 bps |
| + 26 bps taker fee/side | −51.7 bps | −51.7 bps |

The high AUC and the tiny per-trade edge are consistent, and reconciling them
is the whole point: imbalance predicts *direction* well, but the predicted
move is a fraction of a basis point (often a single tick, and most 5s windows
don't move at all). A retail taker pays ~52 bps round trip to capture a third
of a basis point.
**The signal is real and nearly mechanical; the profit is not.** Short-horizon
edge of this kind is only harvestable by market makers who *earn* the spread
instead of paying it — which is exactly why latency and queue position matter
in real markets.

## Cross-asset lead-lag

`leadlag.py` asks whether BTC and ETH inform each other. Their 1-second returns
have a contemporaneous correlation of ~0.38, and the cross-correlation is
asymmetric: ETH's return tracks BTC's *next* second more than the reverse
(−1s: +0.15, −2s: +0.10, −3s: +0.06, versus +0.07 at +1s). So over this sample
**ETH leads BTC by a second or two** — the opposite of the usual "BTC leads
everything" assumption.

That relationship carries no predictive power, though: adding the leader's
return, order flow, and imbalance to the follower's direction model moves AUC
by ~0.00. Two honest reasons — BTC's own AUC is already saturated by the
tick-size effect, so there is no headroom to add, and ETH can't be helped by a
feature (BTC) that lags it. A real lead-lag that carries no independent edge,
and at ~0.15 correlation it would not survive costs anyway. (At one-second
resolution, apparent lead-lag can partly reflect which venue's feed updates
first rather than true information flow — a caveat worth stating.)

## Run it

```bash
pip install -r requirements.txt
python recorder.py                    # leave running; ~sub-GB per day
python book.py                        # replay + validate every checksum
python features.py                    # build 1s bars -> data/features-*.csv
python models.py                      # embargoed walk-forward + feature attribution
python backtest.py --horizon 5        # cost-aware PnL + equity curves
python leadlag.py BTC/USD ETH/USD     # cross-asset lead-lag + cross-feature test
pytest                                # book engine + feature unit tests
```

`tests/data/fixture.jsonl.gz` is a slice of a real recording, so the test
suite validates the engine against genuine exchange checksums.

## What I'd do next

- Maker-side simulation with queue-position modeling (earn the spread
  instead of paying it) — requires L3 or fill data to be honest.
- Extend the cross-asset study to SOL and XRP (now recording) for a full
  correlation matrix and lead-lag hierarchy.
- Deeper features: multi-level OFI and trade-size distributions.
- A C++ book engine for line-rate throughput.

## License

MIT
