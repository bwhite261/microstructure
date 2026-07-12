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
- **Chronological validation only.** `TimeSeriesSplit` — train on the past,
  test on the future. Shuffled CV on autocorrelated series leaks.
- **Direction, not magnitude.** The label is the sign of the forward mid
  move, flat bars dropped; AUC is compared against the 0.5 no-skill baseline.

## Results

Results below are from the current recording session; tables regenerate with
`python models.py && python backtest.py` as more data accumulates
(see `results/metrics.csv` and `results/*.png`).

The consistent picture from the three-way backtest: a real predictive signal
exists at short horizons (AUC meaningfully above 0.5, positive mid-to-mid
PnL), it roughly breaks even against the spread, and taker fees (26 bps/side
on Kraken's base tier) eliminate it entirely. That asymmetry is the point:
short-horizon alpha of this kind is only harvestable by market makers who
earn the spread instead of paying it — which is exactly why speed and queue
position matter in real markets.

## Run it

```bash
pip install -r requirements.txt
python recorder.py                    # leave running; ~sub-GB per day
python book.py                        # replay + validate every checksum
python features.py                    # build 1s bars -> data/features-*.csv
python models.py                      # walk-forward AUC/accuracy table
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
