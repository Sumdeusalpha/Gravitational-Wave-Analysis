Gravitational Wave Constant Lattice Analysis
Expanded Monolithic Pipeline

Purpose

This file provides a monolithic, portable, expanded execution of the gravitational wave constant lattice analysis. It is a strict superset of the original lattice script and exists to expose the full validation, null modeling, and diagnostic surface in a single, auditable file.

This monolith is intended for:

reproducibility checks

forensic validation

reviewer inspection

regression testing

exploratory verification of intermediate stages

It is not intended to replace the original lattice analysis script.

Relationship to the Original Script

The original file:

gw_constant_lattice_analysis.py

is the canonical, focused lattice analysis pipeline. It implements the core ratio extraction and constant matching logic in a disciplined, reviewer-safe form.

The expanded monolith:

gw_constant_lattice_expanded_monolith.py

contains everything the original script does, plus additional layers, including:

extended ON/OFF window analysis

phase-scrambled null surrogates

segment-shuffled null surrogates

Monte Carlo null estimation from observed ON ratios

tolerance robustness sweeps

sliding-window lattice intensity (optional)

classifier-based real vs null validation

raw diagnostic printouts

consolidated CSV and PDF outputs

No logic from the original pipeline is removed or contradicted. The monolith is a strict expansion.

Why a Monolith Exists

The monolithic format serves a specific purpose:

it removes hidden dependencies

it exposes execution order explicitly

it allows reviewers to trace the entire analysis without navigating multiple files

it preserves exploratory diagnostics that are not appropriate for the canonical script

The canonical script should be used for standard runs. The monolith should be used when full transparency is required.

How to Run

Create and activate a Python virtual environment.

Install dependencies listed in requirements.txt.

Run the monolith with Python.

Example:

python gw_constant_lattice_expanded_monolith.py

The script will fetch public GWOSC data as needed and write outputs to the configured output directory.

Portability Notes

The monolith includes a portability shim for output and cache paths.

If a preferred Windows path (for example, an F: drive) exists, it will be used. Otherwise, the script automatically falls back to creating local directories relative to the script location.

Optional environment variables may be used to override defaults:

ASTROPY_CACHE_DIR

GW_OUTPUT_DIR

No code changes are required for portability across systems.

Outputs

The monolith writes all results to the output directory. Typical outputs include:

CSV tables summarizing constant matches

Monte Carlo null statistics

per-event and aggregate scores

classifier performance metrics

optional plots (if enabled)

a PDF summary report

The exact set of outputs depends on configuration toggles inside the script.

Configuration

All configuration is defined at the top of the script and includes:

detector selection

window definitions

null generation counts

tolerance values

visualization toggles

No external configuration files are required.

Interpretation Guidance

This monolithic pipeline is designed to surface structured behavior and stress-test it against multiple null hypotheses. Statistical quantities produced by the script are intended as diagnostic indicators rather than final inferential claims.

Formal interpretation and conclusions are documented separately in the associated paper.

License

This file is distributed under the same license as the rest of the repository.
