import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Simulation parameters
num_pulses = 1_000_000
L = 1.0  # Distance to screen (m)
screen_width = 0.01  # 40 mm wide screen
num_bins = 1000  # Detection resolution
bin_positions = np.linspace(-screen_width / 2, screen_width / 2, num_bins)

# Slit geometry
slit_sep = 1e-3  # 1 mm separation between centers
slit_width = 10e-6  # Each slit 10 microns wide
slit_centers = np.array([-slit_sep / 2, slit_sep / 2])

# Emission parameters
angular_spread = 1e-3  # 1 mrad total angular spread (±0.5 mrad)
position_spread = 0.5e-3  # 0.5 mm spread of emission source

# Create empty histogram
detections = np.zeros(num_bins, dtype=int)

# Simulate pulses
for _ in range(num_pulses):
    # Emit pulse with a slight angular offset and lateral origin shift
    origin_offset = np.random.normal(0, position_spread)
    angle = np.random.normal(0, angular_spread / 2)

    # Propagate to screen
    x_hit = origin_offset + L * np.tan(angle)

    # Check for slit shadowing (passes through aperture geometry)
    through_slit = False
    for center in slit_centers:
        if np.abs(origin_offset - center) <= slit_width / 2:
            through_slit = True
            break

    if not through_slit:
        continue  # Pulse blocked by barrier

    # Bin the detection
    bin_idx = np.argmin(np.abs(bin_positions - x_hit))
    if 0 <= bin_idx < num_bins:
        detections[bin_idx] += 1

# Smooth and plot result
smooth_counts = gaussian_filter1d(detections, sigma=2)
plt.figure(figsize=(10, 4))
plt.plot(bin_positions * 1e3, smooth_counts)
plt.xlabel("Screen position (mm)")
plt.ylabel("Counts")
plt.title("Pulse-Based Double-Slit Detection with Deadtime")
plt.tight_layout()
plt.show()