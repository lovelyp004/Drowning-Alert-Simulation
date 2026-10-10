# Drowning Alert Simulation

Track-level simulation accompanying the paper *Computer Vision for Early Drowning Detection: An Evaluation Protocol and Assumption-Driven Simulation of Accuracy, False Alarms, and Alert Latency in Pools and at Riverbank Ghats* (Lovely Pal).

**This is a simulation. No real video is used.** All results depend on the assumed parameters listed in the paper (Tables 2 and 3) and in `COND` inside the script. They illustrate metric trade-offs and failure mechanisms, not field performance.

## Files

- `drowning_sim.py` - the simulation (main results and sensitivity analysis)
- `sim_results.json` - main results (Tables 4 and 5 of the paper)
- `sens_results.json` - sensitivity results (Tables 6 and 7 of the paper)

## Run

```
pip install numpy numba
python drowning_sim.py main   # writes sim_results.json
python drowning_sim.py sens   # writes sens_results.json
```

Seeds are fixed, so the results regenerate exactly.
