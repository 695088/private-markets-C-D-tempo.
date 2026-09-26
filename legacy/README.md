# Legacy research scripts

Unmodified copies of the upstream IEOR / `priv-equity-distri-contri` modules:

- `PE_Data.py` — load / interpolate / supervised prep
- `PE_LSTM_V5.py` — script-style LSTM train/test
- `PE_MRSR.py` — script-style mean-reverting Monte Carlo

Prefer the `private_markets_cd` package and CLI for new LP C/D tempo work. These files remain as reference and for parity checks against the pluggable backends under `src/private_markets_cd/backends/`.
