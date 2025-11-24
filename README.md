# Gravitational Wave Constant Lattice Analysis

This repository contains the full analysis pipeline used to test the hypothesis that **gravitational wave strain encodes a lattice of mathematical and physical constants**, with structure that **cannot be explained by noise or simple random coincidence**.

Using public LIGO/Virgo/KAGRA events from **GWOSC**, this code:

- Downloads strain data for a large catalog of GW events.
- Extracts **amplitude and time-interval ratios** from local peak/dip structure.
- Matches those ratios against a catalog of **math and physics constants** (π, √2, √3, √5, φ, Feigenbaum α/δ, Khinchin, Planck scales, cosmological scales, etc.).
- Separates:
  - **ON windows** (tight around the GW burst),
  - **OFF windows** (quiet before/after),
  - **null surrogates** (phase-scrambled and segment-shuffled).
- Runs **Monte Carlo bootstraps** on the ON-window ratios.
- Computes **global and per-event Z / χ-like statistics**.
- Trains a **Random Forest classifier** that distinguishes ON vs OFF+null segments using **only constant-match features**, and evaluates:
  - Standard random train/test splits
  - **Event-grouped splits** (train on some events, test on completely different events)
- Evaluates **tolerance robustness**: how the constant lattice behaves as the matching window is widened (e.g. ±0.03, ±0.05, ±0.07).

The result is a **proof engine** for the claim that gravitational wave signals sit on a non-random constant lattice with **φ-bounded, fractal-like growth** of constant matches.

---

## Repository Structure

Suggested layout:

```text
.
├── gw_constant_lattice_analysis.py   # Main analysis script (monolithic)
├── requirements.txt                  # Python dependencies
├── run_analysis.bat                  # Windows helper to run the analysis
└── README.md                         # This file
