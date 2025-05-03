import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from scipy.signal import find_peaks


# Simulation parameters
num_pulses = 1_000_000
L = 1.0
screen_width = 0.01
num_bins = 1000
bin_positions = np.linspace(-screen_width / 2, screen_width / 2, num_bins)

# Slit geometry
slit_sep = 1e-3
slit_width = 10e-6
slit_centers = np.array([-slit_sep / 2, slit_sep / 2])

# Emission parameters
angular_spread = 1e-3
position_spread = 0.5e-3

# Timing parameters
pulse_rate = 1_000_000  # Hz
dark_time = 10e-6  # 10 microseconds
detections = np.zeros(num_bins, dtype=int)

# Initialize time
event_times = np.cumsum(np.random.exponential(1 / pulse_rate, num_pulses))
last_detection_time = -np.inf

# Simulate pulses with Poisson-based timing
for current_time in event_times:
    if current_time - last_detection_time < dark_time:
        continue  # Detector in dark time

    # Emit a pulse
    origin_offset = np.random.normal(0, position_spread)
    angle = np.random.normal(0, angular_spread / 2)
    x_hit = origin_offset + L * np.tan(angle)

    # Slit check
    through_slit = False
    for center in slit_centers:
        if np.abs(origin_offset - center) <= slit_width / 2:
            through_slit = True
            break
    if not through_slit:
        continue

    # Register detection
    bin_idx = np.argmin(np.abs(bin_positions - x_hit))
    if 0 <= bin_idx < num_bins:
        detections[bin_idx] += 1
        last_detection_time = current_time  # Update for dark time tracking

# Plot
smooth_counts = gaussian_filter1d(detections, sigma=2)
plt.figure(figsize=(10, 4))
plt.plot(bin_positions * 1e3, smooth_counts)
plt.xlabel("Screen position (mm)")
plt.ylabel("Counts")
plt.title("Poisson Pulse Detection w/o Dark Time")
plt.tight_layout()
plt.show()

# Smooth the histogram
smooth_counts = gaussian_filter1d(detections, sigma=2)

# Save data
data_df = pd.DataFrame({
    "position_mm": bin_positions * 1e3,
    "counts": detections,
    "smooth_counts": smooth_counts
})

data_path = "double_slit_results.csv"
data_df.to_csv(data_path, index=False)

# Calculate fringe visibility in central region (±2 mm)
center_mask = (bin_positions >= -0.002) & (bin_positions <= 0.002)
central_counts = smooth_counts[center_mask]
central_positions = bin_positions[center_mask]

# Find peaks and valleys
peaks, _ = find_peaks(central_counts, distance=10)
valleys, _ = find_peaks(-central_counts, distance=10)

if len(peaks) > 0 and len(valleys) > 0:
    I_max = np.max(central_counts[peaks])
    I_min = np.min(central_counts[valleys])
    visibility = (I_max - I_min) / (I_max + I_min)
else:
    visibility = None

# Plot result
plt.figure(figsize=(10, 4))
plt.plot(bin_positions * 1e3, smooth_counts)
plt.xlabel("Screen position (mm)")
plt.ylabel("Counts")
plt.title(f"Poisson Pulse Detection with Dark Time\nFringe Visibility: {visibility:.3f}" if visibility else "Pattern Unclear")
plt.tight_layout()
plt.show()
