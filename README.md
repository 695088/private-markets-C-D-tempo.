# private-markets-C-D-tempo

Framework for **private-market LP contributions and distributions** — capital calls (C), distributions (D), and the **tempo** of C vs D across fund life — on Preqin-style Buyout / VC / Real Estate benchmarks.

## Relation to source

Seeded from [695088/priv-equity-distri-contri](https://github.com/695088/priv-equity-distri-contri) (fork of the IEOR 4742 PE cash-flow LSTM project). Original research scripts live under `legacy/`. The product surface is `private_markets_cd`: data → prep → contribution / distribution models → historical pace + forecast tempo → LP commitment API → CLI. LSTM and mean-reverting diffusion are **pluggable backends**, not the top-level API.

## Framework map

```
data/                         Preqin-style .xlsx benchmarks
src/private_markets_cd/
  types.py                    CashFlowKind, FundType, FundSize
  data/load.py                Load C/D series by fund type & size
  data/prep.py                Log-diff, Gaussian densify, supervised windows
  contributions/              Capital-call (Called Up) model wrappers
  distributions/              Distribution model wrappers
  tempo/pace.py               Historical %% Called Up / Distributed tempo
  tempo/schedule.py           Model-backed C/D intensity schedule
  lp.py                       LPCommitment + compare_fund_types
  backends/mrsr.py|lstm.py    Pluggable forecast backends
  cli.py                      summarize|pace|commitment|compare|forecast|tempo
legacy/                       Original PE_Data / PE_LSTM_V5 / PE_MRSR scripts
examples/lp_tempo_demo.py
```

| LP concern | Use this |
|------------|----------|
| Historical call / distribution pace | `build_pace_curve` or `private-markets-cd pace` |
| Scale pace to a $ commitment | `LPCommitment` or `private-markets-cd commitment` |
| Compare Buyout vs VC vs RE | `compare_fund_types` or `private-markets-cd compare` |
| Model-backed C/D forecast | `--backend mrsr\|lstm` via `forecast` / `tempo` |
| Fund type / size dimensions | `FundType` + `FundSize` over `data/*.xlsx` |

**Defaults:** Buyout / size `All` / backend `mrsr`. VC workbooks expose `All` only; Real Estate maps `All-2000` → `All`.

## Quick start

```bash
python3 -m pip install -e ".[dev]"

# Inventory C/D series
private-markets-cd summarize --fund-type buyout

# Historical LP tempo (cumulative % called / distributed)
private-markets-cd pace --fund-type buyout --size All

# Scale to a $10M commitment
private-markets-cd commitment --fund-type buyout --commitment 10000000 --csv /tmp/cf.csv

# Compare fund types for the same commitment
private-markets-cd compare --commitment 10000000

# Optional model-backed intensity forecasts
private-markets-cd forecast --side C --fund-type buyout
private-markets-cd tempo --fund-type buyout --csv /tmp/tempo.csv

python3 examples/lp_tempo_demo.py
```

Optional LSTM backend: `pip install -e ".[lstm]"` then `--backend lstm`.

## Python API

```python
from private_markets_cd import LPCommitment, build_pace_curve, compare_fund_types

lp = LPCommitment(commitment=10_000_000, fund_type="buyout", size="All")
print(lp.overview())
print(lp.cashflows().head())

pace = build_pace_curve("vc", "All")
print(pace.summary())

print(compare_fund_types(commitment=10_000_000))
```

## Tests

```bash
python3 -m pytest -q
```

## Provenance

Academic course code and Preqin-style sample stats are included for modeling research — not production market data. Upstream framing: Buchner, Kaserer & Wagner; LSTM extension from IEOR 4742.
