# private-markets-C-D-tempo

Workspace for **private markets contribution / distribution (C/D) tempo** modeling — forecasting and simulating PE fund cash-flow timing (capital calls vs. distributions).

## Relation to source

Seeded from [695088/priv-equity-distri-contri](https://github.com/695088/priv-equity-distri-contri) (a fork of [chloeymoon/ieor4742-deeplearning-private-equity](https://github.com/chloeymoon/ieor4742-deeplearning-private-equity)): Columbia IEOR deep-learning coursework that predicts **Called Up** (contributions) and **Distributed** cash flows for Buyout, Venture Capital, and Real Estate fund benchmarks using Preqin-style quarterly stats (2014–2019).

This repo keeps the usable modeling slice (Python modules + Excel benchmarks) and reframes it as a starting point for private-markets **C/D tempo** work. IDE metadata, `__pycache__`, and the course PDF deliverable were omitted from the seed.

There is no separate “tempo” module in the source; tempo here means the timing/pace of contributions and distributions over the fund life.

## Stack

- **Python** — data prep and modeling
- **pandas / numpy / openpyxl** — Excel cash-flow series and interpolation
- **PyTorch** — LSTM forecasting (`PE_LSTM_V5.py`)
- **matplotlib** — prediction vs. actual plots
- Mean-reverting diffusion baseline via Monte Carlo (`PE_MRSR.py`)

## Layout

| Path | Role |
|------|------|
| `PE_Data.py` | Load Excel benchmarks; Gaussian stochastic interpolation; supervised windowing (`get_dat`, `interpolate_gaussian`, `prep_dat`). Metric: `"Called Up"` or `"Distributed"`. |
| `PE_LSTM_V5.py` | PyTorch LSTM train/test on distribution (or contribution) series. |
| `PE_MRSR.py` | Ornstein–Uhlenbeck / CIR-style diffusion fit + Euler–Maruyama Monte Carlo. |
| `Buyout_Funds_Stats_2014.xlsx` | Buyout benchmark cash flows (All / Large / Mid / Small sheets). |
| `VC_Funds_Stats_2014.xlsx` | Venture Capital benchmarks. |
| `RE_Stats_benchmarks_2014.xlsx` | Real Estate benchmarks. |
| `requirements.txt` | Runtime dependencies. |

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# LSTM on distributions (default in script: Buyout + "Distributed")
python PE_LSTM_V5.py

# Mean-reverting simulation on contributions (default: Buyout + "Called Up")
python PE_MRSR.py
```

Edit the `file` / `metric` arguments near the top of each script to switch fund type (Buyout / VC / RE) and C vs D.

**Note:** Scripts expect the `.xlsx` files in the working directory. Training can be slow without a GPU; plots open via matplotlib.

## Provenance

- Upstream research framing: Buchner, Kaserer & Wagner (stochastic PE cash-flow modeling), extended with LSTM in the IEOR 4742 deliverable.
- Do not push changes back to the source fork unless intentionally coordinating with that owner.
