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

Numbers below are from a 105-hour recording (190k one-second bars each for BTC
and ETH; 86k each for SOL and XRP, which were added later). Everything
regenerates as the recorder accumulates more data — the tables below reproduced
to within ~0.01 AUC across three runs on 34h, 79h, and 105h of data.

**Direction prediction** (5s horizon, embargoed walk-forward, logistic):

| symbol | AUC | book features only | imb1 alone | last return alone |
|---|---|---|---|---|
| BTC/USD | 0.91 | 0.91 | 0.90 | 0.55 |
| SOL/USD | 0.64 | 0.64 | 0.64 | 0.52 |
| ETH/USD | 0.59 | 0.60 | 0.58 | 0.53 |
| XRP/USD | 0.53 | 0.54 | 0.54 | 0.51 |

Two things to read here. First, the signal is **order-book imbalance**, not
momentum: across all four assets `imb1` alone reproduces nearly the full AUC
while the last return alone sits at ~0.5. Second, the AUC ordering tracks
**tick size**. BTC/USD moves in $0.10 ticks on a ~$65k price — an extremely
coarse grid — so its mid only changes when a best level clears, and best-level
imbalance mechanically telegraphs which side is about to go. The finer-grained
alts land at 0.53–0.64, in line with the microstructure literature. BTC is the
outlier because of its price grid, not because it is more predictable in any
useful sense.

**But is it tradeable?** The three-way backtest (5s, threshold 0.6):

| scenario | BTC | ETH | SOL | XRP |
|---|---|---|---|---|
| mid-to-mid (no cost) | +0.26 | +0.36 | +0.25 | +0.30 |
| crossing the spread | +0.24 | +0.20 | −1.40 | −0.17 |
| + 26 bps taker fee/side | −51.8 | −51.8 | −53.4 | −52.2 |

*(bps per trade, 1k–7.7k trades per symbol)*

The high AUC and the tiny per-trade edge are consistent, and reconciling them
is the whole point: imbalance predicts *direction* well, but the predicted move
is a fraction of a basis point (often a single tick, and most 5s windows don't
move at all). A retail taker pays ~52 bps round trip to capture a third of a
basis point. Note the middle row: on the wider-spread alts the edge is already
gone before fees are applied at all.
**The signal is real and nearly mechanical; the profit is not.** Short-horizon
edge of this kind is only harvestable by market makers who *earn* the spread
instead of paying it — which is exactly why latency and queue position matter
in real markets.

## Cross-asset lead-lag

`leadlag.py` asks whether the four assets inform each other. Two layers.

**They are strongly synchronized.** Contemporaneous 1-second return
correlations run 0.70–0.83 (ETH–SOL tightest at 0.83), and *every* pair's
cross-correlation peaks at lag 0 — one dominant market factor.

**Underneath that is a faint but perfectly consistent tilt.** Comparing
correlation at positive vs. negative lags gives the same leader in all six
pairs, and the ordering is transitive:

| pair | ρ (lag 0) | leader | lead score |
|---|---|---|---|
| BTC–SOL | 0.76 | SOL | 0.034 |
| BTC–ETH | 0.73 | ETH | 0.029 |
| BTC–XRP | 0.70 | XRP | 0.024 |
| ETH–SOL | 0.83 | SOL | 0.014 |
| ETH–XRP | 0.77 | XRP | 0.006 |
| SOL–XRP | 0.75 | SOL | 0.004 |

which resolves to **SOL → XRP → ETH → BTC**. BTC is the follower in all three
of its pairs; SOL leads all three of its. Six independent pair tests agreeing
(and reproducing across separate runs) says the tilt is real.

It is also weak and almost certainly **mechanical rather than informational**.
The scores are 0.4–3.4% asymmetries on top of a lag-0 peak, and leadership
runs inversely to tick coarseness: BTC's discrete mid only updates when a
level clears, so it necessarily trails assets whose mids move continuously.
Consistent with that, cross-asset features buy nothing — adding SOL's return,
order flow, and imbalance to BTC's direction model moves AUC by −0.000.
A real lead-lag ordering that carries no independent edge. (At one-second
resolution, apparent lead-lag can also reflect which feed updates first rather
than true information flow — a caveat worth stating.)

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
- Isolate the tick-size effect directly: re-run the labels on moves of at
  least N ticks, and check whether BTC's AUC collapses toward the alts'.
- Deeper features: multi-level OFI and trade-size distributions.
- A C++ book engine for line-rate throughput.

## License

MIT
