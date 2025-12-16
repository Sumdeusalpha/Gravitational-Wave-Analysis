Gravitational Wave Constant Lattice Analysis

This repository contains a reproducible pipeline for testing whether gravitational-wave strain ratio structure exhibits statistically meaningful alignment with a broad lattice of mathematical, physical, and astronomical constants, beyond what would be expected from noise or simple randomized surrogates.

The analysis uses public LIGO/Virgo/KAGRA events via GWOSC and evaluates constant “hits” using ratio transforms derived from the strain time series (peak-to-peak, peak-to-dip, dip-to-dip; amplitude and time domains), with multiple validation layers including ON/OFF windows, null surrogates, Monte Carlo estimation, tolerance robustness sweeps, and classifier validation.

Repository Contents

gw_constant_lattice_analysis.py
Primary executable pipeline for the lattice analysis.

requirements.txt
Python dependencies required to run the analysis.

Data Source
Documentation describing data provenance, GWOSC usage, and expected runtime behavior.

What the Script Does (High-Level)
Plain-language overview of the pipeline logic from ingestion to outputs.

Reproducibility and Validation
Reproducibility expectations, validation rationale, and interpretation guidance.

gw_constant_analysis_outputs
Output directory where CSVs, plots, and reports are written during execution.

Quick Start

Create a virtual environment.

Install dependencies from requirements.txt.

Run the script using Python.

The script will fetch event data from GWOSC as needed and write all outputs to the configured output directory.

Configuration Notes

Key parameters inside the script control visualization generation, sliding-window analysis, detector selection, window definitions, null generation, and constant-matching tolerances. These can be adjusted without changing the core logic.

Outputs

Typical outputs include CSV tables summarizing constant matches, Monte Carlo statistics, per-event scores, classifier performance metrics, and a PDF summary report. See “What the Script Does (High-Level)” for details.

License

This project is released under the Apache 2.0 license.
