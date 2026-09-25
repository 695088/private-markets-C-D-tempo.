# private-markets-C-D-tempo

Framework for **private-market LP contributions and distributions** — capital calls (C), distributions (D), and the **tempo** of C vs D across fund life — built on Preqin-style Buyout / VC / Real Estate benchmarks.

## Relation to source

Seeded from [695088/priv-equity-distri-contri](https://github.com/695088/priv-equity-distri-contri) (fork of the IEOR 4742 PE cash-flow LSTM project). Original research scripts live under `legacy/` for reference. The product surface is the `private_markets_cd` package: data → prep → contribution / distribution models → tempo schedule → CLI. LSTM and mean-reverting diffusion are **pluggable backends**, not the top-level API.

## Framework map

```
data/                         Preqin-style .xlsx benchmarks
src/private_markets_cd/
  types.py                    CashFlowKind, FundType, FundSize
  data/load.py                Load C/D series by fund type & size
  data/prep.py                Log-diff, Gaussian densify, supervised windows
  contributions/              LP capital-call (Called Up) models
  distributions/              LP distribution models
  tempo/schedule.py           Align C & D → pacing / DPI-proxy schedule
  backends/
    mrsr.py                   Mean-reverting Monte Carlo (default)
    lstm.py                   Optional PyTorch LSTM
  cli.py                      summarize | forecast | tempo
legacy/                       Original PE_Data / PE_LSTM_V5 / PE_MRSR scripts
examples/lp_tempo_demo.py     End-to-end LP tempo demo
```

| LP concern | Module |
|------------|--------|
| Contributions (capital calls) | `contributions` → metric `"Called Up"` |
| Distributions | `distributions` → metric `"Distributed"` |
| Timing / tempo of C vs D | `tempo.build_tempo_schedule` |
| Fund type / size | `FundType` + `FundSize` over `data/*.xlsx` |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Inventory C/D series for buyout benchmarks
private-markets-cd summarize --fund-type buyout

# Forecast contributions or distributions (default backend: mrsr)
private-markets-cd forecast --side C --fund-type buyout --size All
private-markets-cd forecast --side D --fund-type buyout --size All

# Aligned C vs D tempo schedule (optional CSV)
private-markets-cd tempo --fund-type buyout --size All --csv /tmp/tempo.csv

# Or run the example script
python examples/lp_tempo_demo.py
```

Optional LSTM backend:

```bash
pip install -e ".[lstm]"
private-markets-cd forecast --side D --backend lstm --fund-type buyout
```

## Python API

```python
from private_markets_cd import build_tempo_schedule
from private_markets_cd.contributions import forecast_contributions
from private_markets_cd.distributions import forecast_distributions

c = forecast_contributions("buyout", size="All", backend="mrsr", n_paths=200)
d = forecast_distributions("buyout", size="All", backend="mrsr", n_paths=200)
schedule = build_tempo_schedule("buyout", size="All", backend="mrsr")
print(schedule.summary())
print(schedule.to_frame().head())
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## Provenance / license note

Academic course code and Preqin-style sample stats are included for modeling research. Do not treat workbook figures as production market data. Upstream framing: Buchner, Kaserer & Wagner (stochastic PE cash flows), extended with LSTM in IEOR 4742.
