# Coin Flip Trading System

A trading system whose entry signal is a fair coin — heads long, tails short — used
as a **calibrated null hypothesis** for testing whether filters, features, machine
learning and risk management can turn a zero-edge signal into a profitable one.

The short answer, after 15.6 years of 1-minute data and 73 falsifiable tests, is
**no**. This repository is the record of establishing that carefully rather than
assuming it.

---

## Why a coin flip

Most backtests fail because the researcher cannot tell edge from noise. Starting
from a signal with *known* expectancy removes that ambiguity:

- A fair coin has **exactly zero** expected edge before costs and a **strictly
  negative** one after them.
- Therefore any backtest that shows the raw flip making money has a bug — the
  coin is a tripwire for look-ahead bias, survivorship, and cost-modelling errors.
- And any filter layered on top has a benchmark that cannot be argued with: it
  must beat the unfiltered coin, and it must beat trading its own signal without
  the coin.

Every result below is reported against those two benchmarks.

---

## The strategy

| | |
|---|---|
| **Signal** | heads → long (+1), tails → short (−1) |
| **Entry** | one hour before the cash open (08:30 ET target; first bar in [08:30, 09:30)) |
| **Size** | 1 unit — one E-mini contract, one share, or one coin |
| **Stop** | 2% of the account loss limit ($45,000 × 2% = $900) |
| **Exit** | whichever comes first: the stop, or the 16:00 ET cash close |
| **Overnight** | never held |

The flip is a pure function of `(seed, asset, bar timestamp)` via a `blake2b` +
`splitmix64` hash, not a sequential RNG draw. This means a bar's flip cannot
change when the sample is truncated, extended or reordered, and a live session
recomputes the same flip for the same bar without replaying history — so
backtest, paper and live agree by construction.

---

## Architecture

The notebook is organised into modules, each a numbered sequence of "cellblocks".

| Module | Cellblocks | Purpose |
|---|---|---|
| **1 — Dependencies** | 1–3 | imports; local data path; yfinance loader (1h / 720d / pre-post) |
| **2 — Load assets** | 1–11 | /NQ continuous front-month from a local 1m CSV, plus ten assets at 1h |
| **3 — Strategy** | 1–10 | coin-flip signals; entry/exit/risk engine; P&L charts; data layer; resampling |
| **4 — Features** | 1–4 | multi-day slope regime; pre-open EMA gate; results; head-to-head |
| **5 — Bootstrapping** | 1–2 | risk-management rules over resampled trade sequences; charts |
| **6 — Machine learning** | 1–2 | walk-forward random forest gate; classifier performance charts |

### Separation of concerns

- **Module 3 cellblock 1** decides *which way* to bet and nothing else. It emits
  `signal` (the decision a bar produces) and `position` (the shifted, tradable
  version). Downstream P&L must read `position`, never `signal` — acting on a
  bar's own signal trades a bar that has not finished happening.
- **Module 3 cellblock 2** decides *how much*, *where the trade dies*, and *when
  the account stops trading*. It carries a three-layer risk book: per-trade stop,
  daily loss limit, and an account-level kill switch on drawdown from peak.
- **Modules 4 and 6** produce *gates*. Every gate shares one interface — a frame
  with a `trend` column of +1/−1/0 — and is applied by the same `gate_signals()`.
  Gates are strictly **subtractive**: they can remove a trade, never reverse one.

---

## Data

| Source | Coverage | Notes |
|---|---|---|
| Local `/NQ` 1-minute CSV | **5,288,241 bars, 2010–2026 (15.6 yrs)** | continuous front-month, rolled on daily volume with a no-backward-roll constraint; 64 roll events |
| yfinance 1h | ~720–1,050 days, 10 assets | /NQ /ES /RTY /YM, BTC ETH, TSLA NVDA AAPL, ^VIX, VIXY |
| yfinance 1d | 9–46 yrs | kept for regime work; **cannot** be traded by this strategy (see below) |

The local 1m series is the only sample long enough to conclude anything. The
Yahoo 1h sets span about two years each and are treated accordingly.

### Data defects found and handled

These are documented because each one silently produced a wrong result before it
was caught:

1. **Split-day price corruption.** On NVDA's 10-for-1 split date (2024-06-10) the
   pre-market bars carry the *pre-split* price in the wick: `open 120.55,
   close 120.38, high 1208.88`. A short's stop sat $900 away, the bogus high
   "hit" it, and the backtest booked a **−$900.01 loss on a share that never
   traded above $123 that day** — 104% of NVDA's entire net loss came from that
   one bar. Fixed by clamping high/low to the bar's body on **zero-volume** bars
   only. The threshold is measured, not guessed: across all eleven assets every
   wick beyond 1.2× the bar's own body sits on a zero-volume bar, and not one
   bar that actually traded exceeds it (futures and crypto top out at 1.13×).

2. **Equities had no pre-open bar.** Yahoo's regular-hours 1h bars for a stock
   start at 09:30, so nothing existed in the 08:30–09:30 entry window and every
   equity session was skipped. `prepost=True` supplies the 04:00–09:30 bars.
   Verified to be a **no-op** for all seven futures/crypto tickers and ^VIX.
   Caveat: Yahoo reports pre/post **volume as 0**, so liquidity is not
   measurable there and a 1-tick slippage assumption is optimistic.

3. **Daily bars are tz-naive.** yfinance returns naive timestamps for `1d` and
   tz-aware for intraday. A naive index cannot be placed on a wall-clock trading
   session, so daily sets are excluded from the grids by name rather than
   silently, and `run_session_trades` refuses them with an explicit error rather
   than guessing a timezone — a wrong guess would shift every entry by hours and
   still produce a confident-looking equity curve.

4. **Silent upsampling.** `resample_ohlcv`'s guard originally trusted a
   caller-supplied `src_step`, and every batch cell passed the same constant.
   Handing it 1h bars with `src_step="1min"` raised nothing and returned 13,691
   "15m" bars that were really hourly bars wearing a 15m label. The guard now
   **measures** the source step off the index; a declared step can only make the
   check stricter, never looser.

5. **`^VIX` is an index.** You cannot hold one unit of it. It stays as an
   untradable reference and **VIXY** (the short-term VIX futures ETF) is the
   tradable proxy.

---

## Results

### The raw coin flip

/NQ, 1-minute data, 15.6 years, 2,437 trades:

```
net            -$34,115        (-$14.00 per trade)
win rate        46.0%          payoff 1.108
worse than -1R  405 trades (16.6%)   <- gaps beat the stop
kill switch     TRIPPED 2019-12-02
```

Across 20 different coins on the same bars: median **−$34,835**, range −$45,455
to +$169,830, and **16 of 20 blew the $45,000 account**. That spread on identical
data is the single most important number in this repository — it is why no result
here is quoted from one seed.

### Feature gates

Three gates were built, each using different information:

| Gate | Signal |
|---|---|
| `ts_gate` | t-statistic of an OLS slope on 40 days of daily closes |
| `ema_gate` | pre-open EMA(20) vs yesterday's EMA at the cash close |
| `rf_gate` | walk-forward random forest over 12 causal features |

Head-to-head over 10 coins × 11 datasets (110 paired cells, same bars, same coin
on both sides):

```
          cells  per_trade  t_stat  killed_%
raw         110      -0.22   -0.30      29.0
ts_gate     110      -6.27   -0.85      17.0
ema_gate    110      -0.54   -0.77      17.0
ts_only     110     -30.51   -1.48      55.0
ema_only    110      -0.44   -1.42      45.0

P(row beats column), % of 110 paired cells:
           raw  ts_gate  ema_gate  ts_only  ema_only
raw        NaN     83.0      73.0     79.0      65.0
ts_gate   17.0      NaN      35.0     65.0      46.0
ema_gate  27.0     65.0       NaN     73.0      64.0
```

- **The unfiltered coin flip beats every filter.** Raw wins 8 of 11 datasets.
- **EMA is the better of the two price features** (65/35 over the slope gate).
- **Nothing clears the noise band.** Median |t| runs 0.30–1.48 across all arms.
- **Filters do reduce ruin**: killed drops from 29% to 17%. They trade less and
  die less. They just do not make money doing it.

A structural note that applies to every gate: because a gate only removes trades
where the coin *disagrees* with the feature, every surviving trade points the way
the feature already pointed. The coin stops choosing direction and becomes a
random 50% sampler of the feature's signals. That is why `*_only` benchmarks —
the feature traded every session with the coin ignored — are reported beside
every gated arm.

### Machine learning

Random forest, 12 causal features, expanding-window walk-forward, 3,403
out-of-sample sessions:

```
label      54.3% up sessions  ->  base rate 54.3%
accuracy   50.8%              AUC 0.5083
calls      827 up / 866 down / 1,710 unsure
```

**The classifier has no skill** — it scores *below* the base rate, and its ROC
curve is visually indistinguishable from the diagonal along its entire length.
Calibration is flat: 50% of all predictions fall inside a ±0.02 no-call band.
Feature importances are near-uniform (0.123 down to 0.020 against an equal share
of 0.083).

Asked directly what separated the best trades from the worst — top vs bottom
decile by realized P&L, across all 12 conditions — the largest separation is
**0.19 standard deviations**. The conditions are the same on both ends. There is
no condition in this feature set to find, which is consistent with three
independent methods failing on it.

### Risk management

Bootstrapped over 2,000 resampled trade sequences (block bootstrap, block=20, so
losing runs stay intact):

```
Kelly f* = -0.0266   <- NEGATIVE: the optimal stake is ZERO
expectancy = -0.0156 R/trade = -$14.00 at $900 risk
```

| rule | ruin % | med ret | CVaR5 | med DD |
|---|---|---|---|---|
| `pct_0.5` | 0.0 | −17.9% | −43.6% | 29% |
| `dd_scaled` | 0.0 | −19.6% | −44.7% | 32% |
| `pct_1` | 0.05 | −35.0% | −69.2% | 54% |
| `pct_2` | 28.1 | −61.1% | −90.5% | 90% |
| **`fixed_1u`** *(the system's current rule)* | **68.7** | −71.3% | −99.8% | 100% |

Three findings:

1. **Kelly answers the question directly.** Fed 2,437 real trades, `f* = −0.0266`.
   A negative fraction means the optimal stake is zero. Everything else is "given
   that you are going to trade a negative-edge system anyway, how do you lose
   slowest".
2. **Fixed-dollar sizing is the worst rule tested.** It does not shrink when
   equity does, so a losing run walks the account straight into the limit. Every
   fractional rule ruins less than 0.1%.
3. **Smaller is monotonically better**, which is what negative expectancy looks
   like. There is no interior optimum; the limit is zero.

A practical constraint worth knowing: at a $45,000 account with a $900 unit,
1% fractional sizing rounds to **zero whole contracts**. Fixed-fractional sizing
is unavailable at this size without micros (/MNQ at $2/point → ~$90 per unit).

---

## Verification

The notebook is not a collection of plots — every cellblock that makes a claim
ships a falsifiable self-test that raises on failure. **73 properties across 32
code cells.** The ones that matter most:

- **`flips unchanged when every price is rewritten`** — proves the coin cannot
  peek at the future, rather than asserting it.
- **`a session's regime ignores every bar inside that session`** — rewrites an
  entire session by 30% and requires the regime to be unmoved.
- **`shuffled labels score at chance (49.8% vs base 50.8%)`** — the leakage
  detector for the ML pipeline. Identical mechanics, permuted labels; if accuracy
  had stayed high, every other ML number here would be worthless.
- **`stop fills never better than the stop`** — a gap fills worse, not free.
- **`equity endpoint == sum of net P&L`** — charts show the trades, not a
  smoothed version of them.
- **`Kelly f* on W=0.55, payoff=1.5 is 0.25`** — pinned to a published worked
  example.
- **`rolling t-stat matches np.polyfit (max err 6.79e-09)`** and
  **`ROC AUC matches sklearn`** — fast closed-form paths pinned to references.

Data integrity is checked separately: volume conservation across every resampled
timeframe, no duplicate or unsorted timestamps, OHLC range validation, and a
no-lookahead spot check that every 15m bar was built only from 1m bars inside its
own window. All pass on all ten assets.

### Honest engineering notes

- OHLC bars cannot say whether the low or the high came first inside a bar. When
  a bar could have hit the stop, the code **assumes it did**. That is the
  pessimistic reading and the only honest one at this resolution.
- A stop is not a guarantee. Worst realized losses reached −$3,695 against $900
  planned. `planned_risk` is the intent; the reports print realized beside it.
- Charts never drop a dataset silently. An asset that cannot trade gets a
  labelled panel stating *why* — "no bar in the hour before the open" reads as
  "this data cannot answer the question", which is the truth, whereas an absent
  panel reads as "no result".
- Panels do not share an x-axis where the datasets do not share a period
  (/NQ 1m is 15.6 years, the Yahoo sets ~2), and each panel carries its own span.

---

## Running it

### Requirements

Python 3.13 with:

```
numpy pandas matplotlib statsmodels yfinance scikit-learn pyarrow
```

### Setup

The local `/NQ` 1-minute CSV path is set in **Module 1, cellblock 2** and is
machine-specific — edit `file_path` before running. Everything else downloads
from Yahoo at run time.

Run the cells in order. Module 2 cellblock 1 reads the ~0.74 GB CSV (~8s with
`pyarrow`); Module 3 cellblock 4 fetches 1d/1h/1m for every asset (~40s); Module
4 cellblock 4 sweeps 10 coins × 5 arms × 11 datasets (~60s). A full top-to-bottom
run is roughly 4 minutes.

### Reading the output

Read the **dispersion charts first**. A single equity curve shows what one coin
happened to do; the strategy is a random variable, and the fan of 50 coins is the
only chart that says whether any of it means anything.

---

## What would actually be worth trying next

Everything tested here is derived from price, and price has now failed through
three independent methods — a statistical trend test, an EMA, and a random
forest. Another model on the same twelve columns will not help. Genuinely new
information would be:

- order flow / volume profile / book imbalance
- session-type classification (trend day vs balance day)
- walk-forward on the one lead that survived: `ema_only` on /NQ 1m returned
  +$17.21/trade over 3,665 trades and was the only arm in the entire project
  that survived 15.6 years without hitting the kill switch. It is one asset, one
  span, one deterministic path — fit the span on 2010–2018 and trade 2019–2026
  untouched before believing any of it.

---

## Caveats

- **No strategy here is profitable, and none is claimed to be.** Past backtest
  performance does not predict future returns.
- A negative result honestly found is more valuable than a flattering bug. Most
  of this repository is negative results.
- Costs are modelled — commission per side, notional basis-point fees for
  crypto, and 1 tick of adverse slippage on entry and exit — but they are
  *assumptions*. Verify every contract specification and fee against your own
  broker's statement before believing any P&L figure.
- Bootstrap paths assume a fill at any size. Real size moves the market.
- Nothing here has been paper-traded, let alone traded live. It would need
  independent testing, broker sandbox validation, and a tested kill switch before
  any capital is at risk.
- **This is not financial advice.**

## License

See `LICENSE`.
