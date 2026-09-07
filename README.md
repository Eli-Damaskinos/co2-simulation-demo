# CO₂ Storage Resilience Simulator

A Streamlit time-step simulation for testing CO₂ production, storage tanks,
trailer collections and alternative infrastructure configurations.

## Features

- Default production rate plus manual lower-rate or zero-production periods
- Reproducible random downtime blocks covering a requested percentage of time
- Any number of tanks with different capacities, minimum heels and starting levels
- Editable recurring call opportunities, defaulting to 08:00 and 18:00 daily
- A trailer is called only when one certified tank holds its complete usable load
- Manual no-delivery dates and reproducible random missed-delivery percentages
- Configurable trailer target, pump rate and simultaneous loading bays
- Named stresses including four-day gaps, back-to-back trailers and 24-hour gaps
- Monte Carlo comparison of multiple tank, pump and loading-bay configurations
- Operational resilience ranking, mass-balance checking and downloadable results

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Test

```bash
python -m unittest -v test_simulation.py
```

## Ranking interpretation

The comparison ranks physical resilience by lowest worst-case CO₂ loss, then
average loss, pending trailers and storage capacity. It is not a financial
ranking; CAPEX and OPEX inputs are required before comparing economic value.
