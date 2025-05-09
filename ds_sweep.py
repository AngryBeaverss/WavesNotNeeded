import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from itertools import product

# --- Simulation Parameters ---
num_pulses = 500_000
L = 1.0  # propagation distance
screen_width = 0.01  # 1 cm
num_bins = 1000
bin_positions = np.linspace(-screen_width / 2, screen_width / 2, num_bins)
position_spread = 0.5e-3
pulse_rate = 1_000_000  # Hz

# --- Parameter Sweep Ranges ---
angular_spreads = [0.0005, 0.001, 0.002]  # radians
slit_separations = [0.5e-3, 1e-3, 2e-3]  # meters
dark_times = [0, 5e-6, 10e-6, 20e-6]  # seconds

# --- Results Storage ---
results = []

# --- Main Sweep ---
for angular_spread, slit_sep, dark_time in product(angular_spreads, slit_separations, dark_times):
    slit_width = 10e-6
    slit_centers = np.array([-slit_sep / 2, slit_sep / 2])
    detections = np.zeros(num_bins, dtype=int)

    # Poisson timing
    event_times = np.cumsum(np.random.exponential(1 / pulse_rate, num_pulses))
    last_detection_time = -np.inf

    for current_time in event_times:
        if current_time - last_detection_time < dark_time:
            continue

        origin_offset = np.random.normal(0, position_spread)
        angle = np.random.normal(0, angular_spread / 2)
        x_hit = origin_offset + L * np.tan(angle)

        # Check slit passage
        if not any(np.abs(origin_offset - center) <= slit_width / 2 for center in slit_centers):
            continue

        bin_idx = np.argmin(np.abs(bin_positions - x_hit))
        if 0 <= bin_idx < num_bins:
            detections[bin_idx] += 1
            last_detection_time = current_time

    # Analyze visibility
    smooth_counts = gaussian_filter1d(detections, sigma=2)
    center_mask = (bin_positions >= -0.002) & (bin_positions <= 0.002)
    central_counts = smooth_counts[center_mask]
    central_positions = bin_positions[center_mask]

    peaks, _ = find_peaks(central_counts, distance=10)
    valleys, _ = find_peaks(-central_counts, distance=10)

    if len(peaks) > 0 and len(valleys) > 0:
        I_max = np.max(central_counts[peaks])
        I_min = np.min(central_counts[valleys])
        visibility = (I_max - I_min) / (I_max + I_min)
    else:
        visibility = np.nan

    results.append({
        "angular_spread_mrad": angular_spread * 1e3,
        "slit_separation_mm": slit_sep * 1e3,
        "dark_time_us": dark_time * 1e6,
        "visibility": visibility
    })

# --- Save & Optional Plot ---
results_df = pd.DataFrame(results)
results_df.to_csv("forge_visibility_sweep.csv", index=False)
print("Sweep complete. Results saved to forge_visibility_sweep.csv")
