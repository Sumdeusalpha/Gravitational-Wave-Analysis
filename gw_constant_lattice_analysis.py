# coding: utf-8

# =======================================
# Env + imports
# =======================================
import os
os.environ["ASTROPY_CACHE_DIR"] = r"F:\astropy_cache"
os.makedirs(r"F:\astropy_cache", exist_ok=True)

from gwosc.api import fetch_event_json
from gwpy.timeseries import TimeSeries
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GroupShuffleSplit
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
import seaborn as sns
from fpdf import FPDF

# =======================================
# Config
# =======================================
ENABLE_VISUALS = False
ENABLE_SLIDING_ANALYSIS = False

OUTPUT_DIR = r"F:\gw_constant_analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DETECTORS = ["H1", "L1"]

# legacy full-window length (seconds)
LEGACY_WINDOW_HALF = 60.0  # gps-60 to gps+60

# focused ON/OFF burst windows (extension)
ON_BEFORE = 0.2
ON_AFTER = 0.1
OFF1_START = -50.0
OFF1_END = -20.0
OFF2_START = 20.0
OFF2_END = 50.0

# null surrogate config
N_PHASE_SURR = 5
N_SHUFFLE_SURR = 5
SHUFFLE_SEGMENT_SECONDS = 0.1

# Monte Carlo (from real ON ratios)
NUM_MC_SIMULATIONS = 1000

# sliding lattice config
SLIDING_WINDOW_LENGTH = 0.2
SLIDING_WINDOW_STEP = 0.05
SLIDING_RANGE_BEFORE = 0.5
SLIDING_RANGE_AFTER = 0.5

# matching tolerance (match your original: ±0.05)
RTOL = 0.0
ATOL = 0.05

# extra tolerances for robustness check
TOLERANCE_SWEEP = [0.03, 0.05, 0.07]

# =======================================
# PDF sanitization (for FPDF Latin-1)
# =======================================
def pdf_sanitize(text: str) -> str:
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "•": "-",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", "ignore").decode("latin-1")

# =======================================
# Event list
# =======================================
event_names = [
    '151008-v1', '151012.2-v1', '151116-v1', '161202-v1', '161217-v1',
    '170208-v1', '170219-v1', '170405-v1', '170412-v1', '170423-v1',
    '170616-v1', '170630-v1', '170705-v1', '170720-v1', '190924_232654-v1',
    '191118_212859-v1', '191223_014159-v1', '191225_215715-v1',
    '200114_020818-v1', '200121_031748-v1', '200201_203549-v1',
    '200214_224526-v1', '200214_224526-v2', '200219_201407-v1',
    '200311_103121-v1', 'GRB051103-v1', 'GW150914-v1', 'GW150914-v2',
    'GW150914-v3', 'GW151012-v1', 'GW151012-v2', 'GW151012-v3',
    'GW151226-v1', 'GW151226-v2', 'GW170104-v1', 'GW170104-v2',
    'GW170608-v1', 'GW170608-v2', 'GW170608-v3', 'GW170729-v1',
    'GW170809-v1', 'GW170814-v1', 'GW170814-v2', 'GW170814-v3',
    'GW170817-v1', 'GW170817-v2', 'GW170817-v3', 'GW170818-v1',
    'GW170823-v1', 'GW190403_051519-v1', 'GW190408_181802-v1',
    'GW190408_181802-v2', 'GW190412-v1', 'GW190412-v2', 'GW190412-v3',
    'GW190412_053044-v4', 'GW190413_052954-v1', 'GW190413_052954-v2',
    'GW190413_134308-v1', 'GW190413_134308-v2', 'GW190421_213856-v1',
    'GW190421_213856-v2', 'GW190424_180648-v1', 'GW190424_180648-v2',
    'GW190425-v1', 'GW190425-v2', 'GW190425_081805-v3',
    'GW190426_152155-v1', 'GW190426_152155-v2', 'GW190426_190642-v1',
    'GW190503_185404-v1', 'GW190503_185404-v2', 'GW190512_180714-v1',
    'GW190512_180714-v2', 'GW190513_205428-v1', 'GW190513_205428-v2',
    'GW190514_065416-v1', 'GW190514_065416-v2', 'GW190517_055101-v1',
    'GW190517_055101-v2', 'GW190519_153544-v1', 'GW190519_153544-v2',
    'GW190521-v1', 'GW190521-v2', 'GW190521-v3', 'GW190521_030229-v4',
    'GW190521_074359-v1', 'GW190521_074359-v2', 'GW190527_092055-v1',
    'GW190527_092055-v2', 'GW190531_023648-v1', 'GW190602_175927-v1',
    'GW190602_175927-v2', 'GW190620_030421-v1', 'GW190620_030421-v2',
    'GW190630_185205-v1', 'GW190630_185205-v2', 'GW190701_203306-v1',
    'GW190701_203306-v2', 'GW190706_222641-v1', 'GW190706_222641-v2',
    'GW190707_093326-v1', 'GW190707_093326-v2', 'GW190708_232457-v1',
    'GW190708_232457-v2', 'GW190719_215514-v1', 'GW190719_215514-v2',
    'GW190720_000836-v1', 'GW190720_000836-v2', 'GW190725_174728-v1',
    'GW190727_060333-v1', 'GW190727_060333-v2', 'GW190728_064510-v1',
    'GW190728_064510-v2', 'GW190731_140936-v1', 'GW190731_140936-v2',
    'GW190803_022701-v1', 'GW190803_022701-v2', 'GW190805_211137-v1',
    'GW190814-v1', 'GW190814-v2', 'GW190814_211039-v3',
    'GW190828_063405-v1', 'GW190828_063405-v2', 'GW190828_065509-v1',
    'GW190828_065509-v2', 'GW190909_114149-v1', 'GW190909_114149-v2',
    'GW190910_112807-v1', 'GW190910_112807-v2', 'GW190915_235702-v1',
    'GW190915_235702-v2', 'GW190916_200658-v1', 'GW190917_114630-v1',
    'GW190924_021846-v1', 'GW190924_021846-v2', 'GW190925_232845-v1',
    'GW190926_050336-v1', 'GW190929_012149-v1', 'GW190929_012149-v2',
    'GW190930_133541-v1', 'GW190930_133541-v2', 'GW191103_012549-v1',
    'GW191105_143521-v1', 'GW191109_010717-v1', 'GW191113_071753-v1',
    'GW191126_115259-v1', 'GW191127_050227-v1', 'GW191129_134029-v1',
    'GW191204_110529-v1', 'GW191204_171526-v1', 'GW191215_223052-v1',
    'GW191216_213338-v1', 'GW191219_163120-v1', 'GW191222_033537-v1',
    'GW191230_180458-v1', 'GW200105-v1', 'GW200105_162426-v2',
    'GW200112_155838-v1', 'GW200115-v1', 'GW200115_042309-v2',
    'GW200128_022011-v1', 'GW200129_065458-v1', 'GW200202_154313-v1',
    'GW200208_130117-v1', 'GW200208_222617-v1', 'GW200209_085452-v1',
    'GW200210_092254-v1', 'GW200216_220804-v1', 'GW200219_094415-v1',
    'GW200220_061928-v1', 'GW200220_124850-v1', 'GW200224_222234-v1',
    'GW200225_060421-v1', 'GW200302_015811-v1', 'GW200306_093714-v1',
    'GW200308_173609-v1', 'GW200311_115853-v1', 'GW200316_215756-v1',
    'GW200322_091133-v1', 'GW230529_181500-v1', 'blind_injection-v1'
]

# =======================================
# Constants and colors (full)
# =======================================
known_constants = {
    # Core mathematical constants
    'Golden Ratio': 1.61803398875,
    'Phi Squared': 1.61803398875 ** 2,
    'Pi': np.pi,
    'Two Pi': 2 * np.pi,
    'Euler’s Number': np.e,
    'Euler–Mascheroni Constant': 0.5772156649015329,
    'Square Root of 2': np.sqrt(2),
    'Square Root of 3': np.sqrt(3),
    'Square Root of 5': np.sqrt(5),
    'Golden Ratio Conjugate': (np.sqrt(5) - 1) / 2,
    'Plastic Constant': 1.3247179572447458,
    'Apéry’s Constant (zeta(3))': 1.2020569031595942,
    'Catalan’s Constant': 0.9159655941772190,
    'Feigenbaum Delta': 4.6692016091029907,
    'Feigenbaum Alpha': 2.5029078750958928,
    'Khinchin’s Constant': 2.6854520010653062,

    # Planck scale constants
    'Planck Length (m)': 1.616255e-35,
    'Planck Time (s)': 5.391247e-44,
    'Planck Mass (kg)': 2.176434e-8,
    'Planck Charge (C)': 1.875545956e-18,
    'Planck Temperature (K)': 1.416808e32,

    # Thermodynamic / statistical constants
    'Absolute Zero (K)': 0.0,
    'Cosmic Microwave Background Temp (K)': 2.725,
    'Boltzmann Constant (J/K)': 1.380649e-23,
    'Avogadro Constant (1/mol)': 6.02214076e23,
    'Gas Constant R (J/(mol·K))': 8.314462618,

    # Fundamental interaction constants
    'Fine-Structure Constant': 1 / 137.035999074,
    'Gravitational Constant (m^3 kg^-1 s^-2)': 6.67430e-11,
    'Coulomb Constant (N m^2 C^-2)': 8.9875517923e9,

    # Quantum constants
    'Planck Constant (J·s)': 6.62607015e-34,
    'Reduced Planck Constant (J·s)': 1.054571817e-34,

    # Particle masses and charge
    'Electron Mass (kg)': 9.10938356e-31,
    'Proton Mass (kg)': 1.67262192369e-27,
    'Neutron Mass (kg)': 1.67492749804e-27,
    'Muon Mass (kg)': 1.883531627e-28,
    'Tau Mass (kg)': 3.16754e-27,
    'Elementary Charge (C)': 1.602176634e-19,

    # Relativistic constants
    'Speed of Light (m/s)': 2.99792458e8,

    # Cosmological constants / scales
    'Hubble Constant (km/s/Mpc)': 67.4,
    'Critical Density (kg/m^3)': 8.5e-27,
    'Baryon-to-Photon Ratio': 6.1e-10,

    # Astronomical scales
    'Astronomical Unit (m)': 1.495978707e11,
    'Parsec (m)': 3.085677581e16,
    'Solar Mass (kg)': 1.98847e30,
    'Solar Radius (m)': 6.9634e8,
    'Earth Mass (kg)': 5.9722e24,
    'Earth Radius (m)': 6.371e6
}

constant_colors = {
    # Core mathematical constants
    'Golden Ratio': 'gold',
    'Phi Squared': 'pink',
    'Pi': 'orange',
    'Two Pi': 'darkorange',
    'Euler’s Number': 'green',
    'Euler–Mascheroni Constant': 'olive',
    'Square Root of 2': 'purple',
    'Square Root of 3': 'brown',
    'Square Root of 5': 'peru',
    'Golden Ratio Conjugate': 'khaki',
    'Plastic Constant': 'chocolate',
    'Apéry’s Constant (zeta(3))': 'coral',
    'Catalan’s Constant': 'orchid',
    'Feigenbaum Delta': 'firebrick',
    'Feigenbaum Alpha': 'indigo',
    'Khinchin’s Constant': 'slateblue',

    # Planck scale constants
    'Planck Length (m)': 'gray',
    'Planck Time (s)': 'dimgray',
    'Planck Mass (kg)': 'darkslategray',
    'Planck Charge (C)': 'lightcoral',
    'Planck Temperature (K)': 'magenta',

    # Thermodynamic / statistical constants
    'Absolute Zero (K)': 'black',
    'Cosmic Microwave Background Temp (K)': 'blue',
    'Boltzmann Constant (J/K)': 'cyan',
    'Avogadro Constant (1/mol)': 'lightseagreen',
    'Gas Constant R (J/(mol·K))': 'seagreen',

    # Fundamental interaction constants
    'Fine-Structure Constant': 'red',
    'Gravitational Constant (m^3 kg^-1 s^-2)': 'darkgreen',
    'Coulomb Constant (N m^2 C^-2)': 'maroon',

    # Quantum constants
    'Planck Constant (J·s)': 'deeppink',
    'Reduced Planck Constant (J·s)': 'hotpink',

    # Particle masses and charge
    'Electron Mass (kg)': 'darkblue',
    'Proton Mass (kg)': 'violet',
    'Neutron Mass (kg)': 'teal',
    'Muon Mass (kg)': 'navy',
    'Tau Mass (kg)': 'mediumvioletred',
    'Elementary Charge (C)': 'darkred',

    # Relativistic constants
    'Speed of Light (m/s)': 'steelblue',

    # Cosmological constants / scales
    'Hubble Constant (km/s/Mpc)': 'goldenrod',
    'Critical Density (kg/m^3)': 'sienna',
    'Baryon-to-Photon Ratio': 'darkcyan',

    # Astronomical scales
    'Astronomical Unit (m)': 'tan',
    'Parsec (m)': 'cadetblue',
    'Solar Mass (kg)': 'yellowgreen',
    'Solar Radius (m)': 'mediumseagreen',
    'Earth Mass (kg)': 'limegreen',
    'Earth Radius (m)': 'forestgreen'
}

# =======================================
# Helpers: data fetch + windows
# =======================================
def fetch_gw_data(event_name, ifo):
    try:
        event_data = fetch_event_json(event_name)
        gps = event_data['events'][event_name]['GPS']
        start = gps - LEGACY_WINDOW_HALF
        end = gps + LEGACY_WINDOW_HALF
        strain = TimeSeries.fetch_open_data(ifo, start, end, cache=True)
        return strain, gps
    except Exception as e:
        print(f"Error fetching data for {event_name} [{ifo}]: {e}")
        return None, None

def slice_window(strain, t_start, t_end):
    if strain is None:
        return None
    mask = (strain.times.value >= t_start) & (strain.times.value <= t_end)
    if not np.any(mask):
        return None
    return TimeSeries(strain.value[mask], times=strain.times.value[mask])

# =======================================
# Helpers: amplitude/time ratios
# =======================================
def compute_amplitude_ratios(strain):
    if strain is None or len(strain.value) == 0:
        return np.array([]), np.array([]), np.array([])
    peaks, _ = find_peaks(strain.value, height=0)
    dips, _ = find_peaks(-strain.value, height=0)
    peak_values = strain.value[peaks]
    dip_values = strain.value[dips]
    p2p = [peak_values[i] / peak_values[i - 1]
           for i in range(1, len(peak_values)) if peak_values[i - 1] != 0]
    p2d = [peak_values[i] / dip_values[i]
           for i in range(min(len(peak_values), len(dips))) if dip_values[i] != 0]
    d2d = [dip_values[i] / dip_values[i - 1]
           for i in range(1, len(dip_values)) if dip_values[i - 1] != 0]
    return np.array(p2p), np.array(p2d), np.array(d2d)

def compute_time_ratios(strain):
    if strain is None or len(strain.value) == 0:
        return np.array([]), np.array([]), np.array([])
    peaks, _ = find_peaks(strain.value, height=0)
    dips, _ = find_peaks(-strain.value, height=0)
    t = strain.times.value
    peak_times = t[peaks]
    dip_times = t[dips]
    dt_peak = np.diff(peak_times)
    dt_dip = np.diff(dip_times)
    p2p = [dt_peak[i] / dt_peak[i - 1]
           for i in range(1, len(dt_peak)) if dt_peak[i - 1] != 0]
    min_len = min(len(peak_times), len(dip_times))
    dt_p2d = []
    for i in range(1, min_len):
        dt_curr = abs(peak_times[i] - dip_times[i])
        dt_prev = abs(peak_times[i - 1] - dip_times[i - 1])
        if dt_prev != 0:
            dt_p2d.append(dt_curr / dt_prev)
    d2d = [dt_dip[i] / dt_dip[i - 1]
           for i in range(1, len(dt_dip)) if dt_dip[i - 1] != 0]
    return np.array(p2p), np.array(dt_p2d), np.array(d2d)

# =======================================
# Helpers: constant matching + features
# =======================================
def count_constant_matches(ratios, rtol=None, atol=None):
    if rtol is None:
        rtol = RTOL
    if atol is None:
        atol = ATOL
    counts = {}
    if ratios is None or len(ratios) == 0:
        for c in known_constants.keys():
            counts[c] = 0
        return counts
    for constant, val in known_constants.items():
        mask = np.isclose(ratios, val, rtol=rtol, atol=atol)
        counts[constant] = int(np.sum(mask))
    return counts

def build_feature_vector_from_ratios(ratio_arrays):
    ratio_arrays = [r for r in ratio_arrays if r is not None and len(r) > 0]
    if len(ratio_arrays) == 0:
        combined = np.array([])
    else:
        combined = np.concatenate(ratio_arrays)
    feat = []
    for val in known_constants.values():
        if len(combined) == 0:
            feat.append(0)
        else:
            match = np.any(np.isclose(combined, val, rtol=RTOL, atol=ATOL))
            feat.append(1 if match else 0)
    return feat

# =======================================
# Helpers: null surrogates
# =======================================
def make_phase_scrambled_surrogate(strain):
    if strain is None or len(strain.value) == 0:
        return None
    y = strain.value
    n = len(y)
    Y = np.fft.rfft(y)
    mags = np.abs(Y)
    phases = np.angle(Y)
    random_phases = np.random.uniform(0, 2 * np.pi, size=len(phases))
    random_phases[0] = phases[0]
    Y_surr = mags * np.exp(1j * random_phases)
    y_surr = np.fft.irfft(Y_surr, n=n)
    return TimeSeries(y_surr, times=strain.times.value)

def make_segment_shuffled_surrogate(strain, segment_seconds=0.1):
    if strain is None or len(strain.value) == 0:
        return None
    fs = float(strain.sample_rate.value)
    seg_len = max(1, int(segment_seconds * fs))
    y = strain.value
    n = len(y)
    n_segs = n // seg_len
    if n_segs <= 1:
        return None
    trimmed = y[:n_segs * seg_len]
    segments = trimmed.reshape(n_segs, seg_len)
    idx = np.random.permutation(n_segs)
    shuffled = segments[idx].reshape(-1)
    if n_segs * seg_len < n:
        tail = y[n_segs * seg_len:]
        shuffled = np.concatenate([shuffled, tail])
    return TimeSeries(shuffled, times=strain.times.value)

# =======================================
# Helpers: visuals (toggled)
# =======================================
def visualize_strain_with_constants(strain, event_name, detector,
                                   amp_ratios, time_ratios):
    if not ENABLE_VISUALS:
        return
    if strain is None or len(strain.value) == 0:
        return
    peaks, _ = find_peaks(strain.value, height=0)
    dips, _ = find_peaks(-strain.value, height=0)
    amp_p2p, amp_p2d, amp_d2d = amp_ratios
    time_p2p, time_p2d, time_d2d = time_ratios
    plt.figure(figsize=(12, 6))
    plt.plot(strain.times.value, strain.value, label='Strain', color='lightgrey', linewidth=0.8)
    ymin = float(np.min(strain.value))
    ymax = float(np.max(strain.value))
    for constant, color in constant_colors.items():
        val = known_constants[constant]
        mt_p2p = [strain.times.value[peaks[i]]
                  for i in range(1, len(peaks))
                  if len(amp_p2p) >= i and np.isclose(amp_p2p[i - 1], val, rtol=RTOL, atol=ATOL)]
        if mt_p2p:
            plt.vlines(mt_p2p, ymin=ymin, ymax=ymax, color=color,
                       linestyle='--', alpha=0.4, label=f'P2P {constant}')
        mt_p2d = [strain.times.value[peaks[i]]
                  for i in range(min(len(peaks), len(dips)))
                  if len(amp_p2d) > i and np.isclose(amp_p2d[i], val, rtol=RTOL, atol=ATOL)]
        if mt_p2d:
            plt.vlines(mt_p2d, ymin=ymin, ymax=ymax, color=color,
                       linestyle=':', alpha=0.6, label=f'P2D {constant}')
        mt_d2d = [strain.times.value[dips[i]]
                  for i in range(1, len(dips))
                  if len(amp_d2d) >= i and np.isclose(amp_d2d[i - 1], val, rtol=RTOL, atol=ATOL)]
        if mt_d2d:
            plt.vlines(mt_d2d, ymin=ymin, ymax=ymax, color=color,
                       linestyle='-.', alpha=0.8, label=f'D2D {constant}')
    plt.title(f"{event_name} [{detector}] Strain with Constant Matches")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{event_name}_{detector}_strain_constants.png")
    plt.savefig(path, dpi=150)
    plt.close()

def plot_feature_distribution(features, constant_names):
    if not ENABLE_VISUALS:
        return
    df = pd.DataFrame(features, columns=[f"{c} Match" for c in constant_names])
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df)
    plt.xticks(rotation=90)
    plt.title("Distribution of Constant Match Features")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_distribution_boxplot.png")
    plt.savefig(path, dpi=150)
    plt.close()

# =======================================
# Sliding lattice intensity (toggled)
# =======================================
def compute_lattice_intensity(strain, gps, event_name, detector):
    rows = []
    if not ENABLE_SLIDING_ANALYSIS or strain is None or len(strain.value) == 0:
        return rows
    t0 = gps - SLIDING_RANGE_BEFORE
    t1 = gps + SLIDING_RANGE_AFTER
    start = t0
    while start + SLIDING_WINDOW_LENGTH <= t1:
        end = start + SLIDING_WINDOW_LENGTH
        window = slice_window(strain, start, end)
        if window is None:
            start += SLIDING_WINDOW_STEP
            continue
        amp_p2p, amp_p2d, amp_d2d = compute_amplitude_ratios(window)
        time_p2p, time_p2d, time_d2d = compute_time_ratios(window)
        ratios = [amp_p2p, amp_p2d, amp_d2d, time_p2p, time_p2d, time_d2d]
        ratios = [r for r in ratios if len(r) > 0]
        combined = np.concatenate(ratios) if len(ratios) > 0 else np.array([])
        total_matches = 0
        if len(combined) > 0:
            for val in known_constants.values():
                total_matches += int(np.sum(np.isclose(combined, val, rtol=RTOL, atol=ATOL)))
        rows.append({
            "Event": event_name,
            "Detector": detector,
            "WindowStart": start,
            "WindowEnd": end,
            "CenterRelToGPS": ((start + end) / 2.0) - gps,
            "TotalMatches": total_matches
        })
        start += SLIDING_WINDOW_STEP
    return rows

# =======================================
# Containers for all results
# =======================================
legacy_rows = []  # full-window (120 s) results

aggregate_counts_on = {c: 0 for c in known_constants.keys()}
per_event_on_counts = []
all_real_on_ratios = []
per_segment_constant_counts = []
sliding_rows_all = []

clf_features = []
clf_labels = []
clf_meta = []
used_real_on_segments = 0

# =======================================
# Main loop: legacy + extended in one pass
# =======================================
for event_name in event_names:
    for ifo in DETECTORS:
        strain, gps = fetch_gw_data(event_name, ifo)
        if strain is None or gps is None:
            continue

        print(f"\n==============================")
        print(f"Data fetched for {event_name} [{ifo}]")

        # ---------- Legacy full-window analysis (120 s) ----------
        amp_p2p_full, amp_p2d_full, amp_d2d_full = compute_amplitude_ratios(strain)

        legacy_p2p_counts = count_constant_matches(amp_p2p_full)
        legacy_p2d_counts = count_constant_matches(amp_p2d_full)
        legacy_d2d_counts = count_constant_matches(amp_d2d_full)

        print(f"\n[LEGACY 120s] Constant Match Summary for {event_name} [{ifo}] (Peak-to-Peak):")
        for c, k in legacy_p2p_counts.items():
            print(f"{c}: {k} matches")

        print(f"\n[LEGACY 120s] Constant Match Summary for {event_name} [{ifo}] (Peak-to-Dip):")
        for c, k in legacy_p2d_counts.items():
            print(f"{c}: {k} matches")

        print(f"\n[LEGACY 120s] Constant Match Summary for {event_name} [{ifo}] (Dip-to-Dip):")
        for c, k in legacy_d2d_counts.items():
            print(f"{c}: {k} matches")

        for constant in known_constants.keys():
            legacy_rows.append({
                "Event": event_name,
                "Detector": ifo,
                "WindowType": "LEGACY_120s",
                "Domain": "AMP_P2P",
                "Constant": constant,
                "Count": legacy_p2p_counts[constant]
            })
            legacy_rows.append({
                "Event": event_name,
                "Detector": ifo,
                "WindowType": "LEGACY_120s",
                "Domain": "AMP_P2D",
                "Constant": constant,
                "Count": legacy_p2d_counts[constant]
            })
            legacy_rows.append({
                "Event": event_name,
                "Detector": ifo,
                "WindowType": "LEGACY_120s",
                "Domain": "AMP_D2D",
                "Constant": constant,
                "Count": legacy_d2d_counts[constant]
            })

        # sliding lattice intensity (optional, fine-grained)
        sliding_rows = compute_lattice_intensity(strain, gps, event_name, ifo)
        sliding_rows_all.extend(sliding_rows)

        # ---------- Extended ON/OFF + null analysis ----------
        on_start = gps - ON_BEFORE
        on_end = gps + ON_AFTER
        off1_start = gps + OFF1_START
        off1_end = gps + OFF1_END
        off2_start = gps + OFF2_START
        off2_end = gps + OFF2_END

        windows = {
            "ON": slice_window(strain, on_start, on_end),
            "OFF1": slice_window(strain, off1_start, off1_end),
            "OFF2": slice_window(strain, off2_start, off2_end)
        }

        for wtype, wstrain in windows.items():
            if wstrain is None or len(wstrain.value) == 0:
                continue

            amp_p2p, amp_p2d, amp_d2d = compute_amplitude_ratios(wstrain)
            time_p2p, time_p2d, time_d2d = compute_time_ratios(wstrain)
            amp_ratios = (amp_p2p, amp_p2d, amp_d2d)
            time_ratios = (time_p2p, time_p2d, time_d2d)

            if wtype == "ON":
                for r in [amp_p2p, amp_p2d, amp_d2d, time_p2p, time_p2d, time_d2d]:
                    if r is not None and len(r) > 0:
                        all_real_on_ratios.append(r)
                used_real_on_segments += 1

            if ENABLE_VISUALS and wtype == "ON":
                visualize_strain_with_constants(wstrain, event_name, ifo, amp_ratios, time_ratios)

            for domain, ratios in [
                ("AMP_P2P", amp_p2p),
                ("AMP_P2D", amp_p2d),
                ("AMP_D2D", amp_d2d),
                ("TIME_P2P", time_p2p),
                ("TIME_P2D", time_p2d),
                ("TIME_D2D", time_d2d),
            ]:
                c_counts = count_constant_matches(ratios)
                for const_name, cnt in c_counts.items():
                    per_segment_constant_counts.append({
                        "Event": event_name,
                        "Detector": ifo,
                        "WindowType": wtype,
                        "Domain": domain,
                        "Constant": const_name,
                        "Count": cnt
                    })
                    if wtype == "ON":
                        aggregate_counts_on[const_name] += cnt
                        per_event_on_counts.append({
                            "Event": event_name,
                            "Detector": ifo,
                            "Constant": const_name,
                            "Count": cnt
                        })

            fv = build_feature_vector_from_ratios(
                [amp_p2p, amp_p2d, amp_d2d, time_p2p, time_p2d, time_d2d]
            )
            clf_features.append(fv)
            clf_labels.append(1 if wtype == "ON" else 0)
            clf_meta.append({
                "Event": event_name,
                "Detector": ifo,
                "WindowType": wtype,
                "SampleType": "REAL_ON" if wtype == "ON" else "REAL_OFF"
            })

        on_strain = windows.get("ON", None)
        if on_strain is None or len(on_strain.value) == 0:
            continue

        # phase-scrambled nulls
        for _ in range(N_PHASE_SURR):
            surr = make_phase_scrambled_surrogate(on_strain)
            if surr is None:
                continue
            amp_p2p, amp_p2d, amp_d2d = compute_amplitude_ratios(surr)
            time_p2p, time_p2d, time_d2d = compute_time_ratios(surr)
            for domain, ratios in [
                ("AMP_P2P", amp_p2p),
                ("AMP_P2D", amp_p2d),
                ("AMP_D2D", amp_d2d),
                ("TIME_P2P", time_p2p),
                ("TIME_P2D", time_p2d),
                ("TIME_D2D", time_d2d),
            ]:
                c_counts = count_constant_matches(ratios)
                for const_name, cnt in c_counts.items():
                    per_segment_constant_counts.append({
                        "Event": event_name,
                        "Detector": ifo,
                        "WindowType": "NULL_PHASE",
                        "Domain": domain,
                        "Constant": const_name,
                        "Count": cnt
                    })
            fv = build_feature_vector_from_ratios(
                [amp_p2p, amp_p2d, amp_d2d, time_p2p, time_p2d, time_d2d]
            )
            clf_features.append(fv)
            clf_labels.append(0)
            clf_meta.append({
                "Event": event_name,
                "Detector": ifo,
                "WindowType": "NULL_PHASE",
                "SampleType": "NULL_PHASE"
            })

        # segment-shuffled nulls
        for _ in range(N_SHUFFLE_SURR):
            surr = make_segment_shuffled_surrogate(on_strain, SHUFFLE_SEGMENT_SECONDS)
            if surr is None:
                continue
            amp_p2p, amp_p2d, amp_d2d = compute_amplitude_ratios(surr)
            time_p2p, time_p2d, time_d2d = compute_time_ratios(surr)
            for domain, ratios in [
                ("AMP_P2P", amp_p2p),
                ("AMP_P2D", amp_p2d),
                ("AMP_D2D", amp_d2d),
                ("TIME_P2P", time_p2p),
                ("TIME_P2D", time_p2d),
                ("TIME_D2D", time_d2d),
            ]:
                c_counts = count_constant_matches(ratios)
                for const_name, cnt in c_counts.items():
                    per_segment_constant_counts.append({
                        "Event": event_name,
                        "Detector": ifo,
                        "WindowType": "NULL_SHUFFLE",
                        "Domain": domain,
                        "Constant": const_name,
                        "Count": cnt
                    })
            fv = build_feature_vector_from_ratios(
                [amp_p2p, amp_p2d, amp_d2d, time_p2p, time_p2d, time_d2d]
            )
            clf_features.append(fv)
            clf_labels.append(0)
            clf_meta.append({
                "Event": event_name,
                "Detector": ifo,
                "WindowType": "NULL_SHUFFLE",
                "SampleType": "NULL_SHUFFLE"
            })

# =======================================
# Save legacy + per-segment + sliding
# =======================================
legacy_df = pd.DataFrame(legacy_rows)
legacy_df.to_csv(os.path.join(OUTPUT_DIR, "legacy_120s_constant_matches.csv"), index=False)

per_seg_df = pd.DataFrame(per_segment_constant_counts)
per_seg_df.to_csv(os.path.join(OUTPUT_DIR, "constant_matches_per_segment.csv"), index=False)

if len(sliding_rows_all) > 0:
    sliding_df = pd.DataFrame(sliding_rows_all)
    sliding_df.to_csv(os.path.join(OUTPUT_DIR, "sliding_lattice_intensity.csv"), index=False)

# =======================================
# PRINT RAW COUNTS BEFORE MONTE CARLO
# =======================================
print("\n=== RAW CONSTANT MATCH COUNTS PER SEGMENT (EXTENDED) ===")
if not per_seg_df.empty:
    grouped = per_seg_df.groupby(["Event", "Detector", "WindowType", "Domain"])
    for (evt, det, wtype, domain), grp in grouped:
        print(f"\nEvent={evt} | Detector={det} | Window={wtype} | Domain={domain}")
        for _, row in grp.iterrows():
            print(f"  {row['Constant']}: {row['Count']}")

print("\n=== AGGREGATE ON-WINDOW COUNTS (EXTENDED, SHORT ON) ===")
for c_name in known_constants.keys():
    print(f"{c_name}: {aggregate_counts_on[c_name]}")

# =======================================
# Monte Carlo null from real ON ratios
# =======================================
all_real_on_ratios = [r for r in all_real_on_ratios if len(r) > 0]
if len(all_real_on_ratios) == 0:
    raise RuntimeError("No ON-window ratios collected; check ON_BEFORE/ON_AFTER or data.")
all_real_on_ratios = np.concatenate(all_real_on_ratios)
num_real_segments = max(1, used_real_on_segments)
avg_ratios_per_segment = max(5, int(round(len(all_real_on_ratios) / num_real_segments)))

mc_per_segment_counts = {c: [] for c in known_constants.keys()}

for _ in range(NUM_MC_SIMULATIONS):
    for _seg in range(num_real_segments):
        sampled_indices = np.random.randint(0, len(all_real_on_ratios), size=avg_ratios_per_segment)
        sim_ratios = all_real_on_ratios[sampled_indices]
        c_counts = count_constant_matches(sim_ratios)
        for c_name, cnt in c_counts.items():
            mc_per_segment_counts[c_name].append(cnt)

mc_stats_rows = []
for c_name, counts in mc_per_segment_counts.items():
    mean_val = float(np.mean(counts))
    std_val = float(np.std(counts))
    mc_stats_rows.append({
        "Constant": c_name,
        "MeanPerSegment": mean_val,
        "StdPerSegment": std_val
    })
mc_stats_df = pd.DataFrame(mc_stats_rows)
mc_stats_df.to_csv(os.path.join(OUTPUT_DIR, "monte_carlo_per_segment_stats.csv"), index=False)
print("\nMonte Carlo per-segment stats:")
print(mc_stats_df)

# =======================================
# Aggregate ON counts + global Z/chi approx
# =======================================
agg_rows = []
chi_rows = []
for c_name in known_constants.keys():
    obs_total = aggregate_counts_on[c_name]
    mean_seg = mc_stats_df.loc[mc_stats_df["Constant"] == c_name, "MeanPerSegment"].values[0]
    std_seg = mc_stats_df.loc[mc_stats_df["Constant"] == c_name, "StdPerSegment"].values[0]
    expected_total = mean_seg * num_real_segments
    std_total = std_seg * np.sqrt(num_real_segments) if std_seg > 0 else 0.0
    if std_total > 0:
        z = (obs_total - expected_total) / std_total
        chi_approx = z ** 2
        p_val = 2 * norm.sf(abs(z))
    else:
        z = np.nan
        chi_approx = np.nan
        p_val = np.nan
    agg_rows.append({
        "Constant": c_name,
        "ObservedTotal": obs_total,
        "ExpectedTotal": expected_total,
        "StdTotal": std_total
    })
    chi_rows.append({
        "Constant": c_name,
        "ZScore_Total": z,
        "ChiSquareApprox_Total": chi_approx,
        "PValueApprox_Total": p_val
    })
agg_df = pd.DataFrame(agg_rows)
chi_df_total = pd.DataFrame(chi_rows)
agg_df.to_csv(os.path.join(OUTPUT_DIR, "aggregate_on_counts.csv"), index=False)
chi_df_total.to_csv(os.path.join(OUTPUT_DIR, "chi_like_global_total.csv"), index=False)

# =======================================
# Per-event Z using per-segment approximation
# =======================================
per_event_on_df = pd.DataFrame(per_event_on_counts)
per_event_z_rows = []
for c_name in known_constants.keys():
    mean_seg = mc_stats_df.loc[mc_stats_df["Constant"] == c_name, "MeanPerSegment"].values[0]
    std_seg = mc_stats_df.loc[mc_stats_df["Constant"] == c_name, "StdPerSegment"].values[0]
    if std_seg == 0:
        continue
    sub = per_event_on_df[per_event_on_df["Constant"] == c_name]
    for _, row in sub.iterrows():
        obs = row["Count"]
        z = (obs - mean_seg) / std_seg
        chi_approx = z ** 2
        p_val = 2 * norm.sf(abs(z))
        per_event_z_rows.append({
            "Event": row["Event"],
            "Detector": row["Detector"],
            "Constant": c_name,
            "ObservedCount": obs,
            "ExpectedPerSegment": mean_seg,
            "StdPerSegment": std_seg,
            "ZScore": z,
            "ChiSquareApprox": chi_approx,
            "PValueApprox": p_val
        })
per_event_z_df = pd.DataFrame(per_event_z_rows)
per_event_z_df.to_csv(os.path.join(OUTPUT_DIR, "per_event_z_scores.csv"), index=False)

# =======================================
# Tolerance robustness table (ON-only)
# =======================================
tol_rows = []
for tol in TOLERANCE_SWEEP:
    for c_name, val in known_constants.items():
        if len(all_real_on_ratios) == 0:
            total_matches = 0
        else:
            total_matches = int(np.sum(np.isclose(all_real_on_ratios, val, rtol=0.0, atol=tol)))
        tol_rows.append({
            "Tolerance": tol,
            "Constant": c_name,
            "TotalMatches": total_matches
        })
tol_df = pd.DataFrame(tol_rows)
# rank within each tolerance (1 = most hits)
tol_df["RankWithinTolerance"] = tol_df.groupby("Tolerance")["TotalMatches"].rank(ascending=False, method="dense")
tol_df.to_csv(os.path.join(OUTPUT_DIR, "tolerance_robustness_on_ratios.csv"), index=False)

# =======================================
# Classifier: real vs null (ON vs OFF+surrogates)
# =======================================
clf_features = np.array(clf_features)
clf_labels = np.array(clf_labels)
clf_meta_df = pd.DataFrame(clf_meta)
clf_meta_df.to_csv(os.path.join(OUTPUT_DIR, "classifier_meta.csv"), index=False)
plot_feature_distribution(clf_features, list(known_constants.keys()))

if clf_features.shape[0] < 5:
    raise RuntimeError("Not enough classifier samples; check data processing.")

indices = np.arange(len(clf_features))
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    clf_features, clf_labels, indices, test_size=0.25, random_state=42
)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
preds = rf_model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"\nRandom Forest REAL vs NULL accuracy: {acc:.4f}")

fi = rf_model.feature_importances_
fi_rows = []
for i, cname in enumerate(known_constants.keys()):
    fi_rows.append({"Constant": cname, "Importance": float(fi[i])})
fi_df = pd.DataFrame(fi_rows)
fi_df.to_csv(os.path.join(OUTPUT_DIR, "classifier_feature_importances.csv"), index=False)

model_path = os.path.join(OUTPUT_DIR, "rf_real_vs_null_model.joblib")
joblib.dump(rf_model, model_path)

test_meta = clf_meta_df.iloc[idx_test].copy()
test_meta["Prediction"] = preds
test_meta["ActualLabel"] = y_test
test_meta.to_csv(os.path.join(OUTPUT_DIR, "classifier_test_predictions.csv"), index=False)

cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
cv_df = pd.DataFrame({"Fold": np.arange(1, len(cv_scores) + 1), "Score": cv_scores})
cv_df.to_csv(os.path.join(OUTPUT_DIR, "classifier_crossval_scores.csv"), index=False)

# =======================================
# Event-grouped robustness: split by event
# =======================================
groups = clf_meta_df["Event"].values
gss = GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
group_cv_rows = []
for split_idx, (train_idx, test_idx) in enumerate(gss.split(clf_features, clf_labels, groups=groups)):
    X_tr, X_te = clf_features[train_idx], clf_features[test_idx]
    y_tr, y_te = clf_labels[train_idx], clf_labels[test_idx]
    rf_g = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_g.fit(X_tr, y_tr)
    preds_g = rf_g.predict(X_te)
    acc_g = accuracy_score(y_te, preds_g)
    group_cv_rows.append({
        "Split": split_idx + 1,
        "Accuracy": acc_g,
        "TrainEvents": len(np.unique(groups[train_idx])),
        "TestEvents": len(np.unique(groups[test_idx]))
    })
group_cv_df = pd.DataFrame(group_cv_rows)
group_cv_df.to_csv(os.path.join(OUTPUT_DIR, "classifier_grouped_event_splits.csv"), index=False)

# =======================================
# PDF summary report
# =======================================
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

title = pdf_sanitize("Gravitational Wave Constant Lattice Analysis")
pdf.cell(200, 10, txt=title, ln=True, align='C')

line = pdf_sanitize(f"Real-vs-Null RF Accuracy: {acc:.4f}")
pdf.cell(200, 10, txt=line, ln=True)

line = pdf_sanitize(f"Mean CV Accuracy (random splits): {cv_scores.mean():.4f}")
pdf.cell(200, 10, txt=line, ln=True)

if not group_cv_df.empty:
    line = pdf_sanitize(
        f"Mean Event-Grouped Accuracy: {group_cv_df['Accuracy'].mean():.4f}"
    )
    pdf.cell(200, 10, txt=line, ln=True)

pdf.cell(200, 10, txt=pdf_sanitize("Top Feature Importances:"), ln=True)

fi_sorted = fi_df.sort_values("Importance", ascending=False)
for _, row in fi_sorted.iterrows():
    const_name = pdf_sanitize(str(row["Constant"]))
    line = pdf_sanitize(f"{const_name}: {row['Importance']:.4f}")
    pdf.cell(200, 8, txt=line, ln=True)

pdf_path = os.path.join(OUTPUT_DIR, "GW_Constant_Lattice_Report.pdf")
pdf.output(pdf_path)
print(f"PDF summary saved to {pdf_path}")
