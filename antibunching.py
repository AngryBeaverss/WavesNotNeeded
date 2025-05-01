import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PARAMETERS
# -----------------------------
num_pulses = 1000                # Total number of pulses
min_separation = 0.2             # Minimum time between pulses (anti-bunching)
jitter_range = (0.05, 0.15)      # Jitter to add variability
quantum_efficiency = 0.7         # Detection probability
window_size = 1.0                # Max τ delay (s)
bin_width = 0.01                 # τ resolution
np.random.seed(42)               # Reproducibility

# -----------------------------
# STEP 1: GENERATE ANTI-BUNCHED PULSES
# -----------------------------
t_emit = [0.0]
while len(t_emit) < num_pulses:
    jitter = np.random.uniform(*jitter_range)
    t_next = t_emit[-1] + min_separation + jitter
    t_emit.append(t_next)
t_emit = np.array(t_emit)

# -----------------------------
# STEP 2: DETECTION VIA BEAMSPLITTER
# -----------------------------
detector_A = []
detector_B = []

for t in t_emit:
    if np.random.rand() < 0.5:
        if np.random.rand() < quantum_efficiency:
            detector_A.append(t)
    else:
        if np.random.rand() < quantum_efficiency:
            detector_B.append(t)

detector_A = np.array(detector_A)
detector_B = np.array(detector_B)

# -----------------------------
# STEP 3: COINCIDENCE HISTOGRAM
# -----------------------------
taus = []
for t1 in detector_A:
    nearby = detector_B[np.abs(detector_B - t1) <= window_size]
    taus.extend(nearby - t1)

taus = np.array(taus)
bins = np.arange(-window_size, window_size + bin_width, bin_width)
hist, edges = np.histogram(taus, bins=bins)
bin_centers = (edges[:-1] + edges[1:]) / 2

# Normalize g²(τ)
g2_tau = hist / np.mean(hist)

# -----------------------------
# STEP 4: PLOT g²(τ)
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(bin_centers, g2_tau, drawstyle='steps-mid', color='darkblue')
plt.axhline(1.0, linestyle='--', color='gray', label='Poissonian baseline (g²=1)')
plt.title("Second-Order Correlation Function g²(τ): Anti-Bunched Pulse Stream")
plt.xlabel("Delay τ (seconds)")
plt.ylabel("g²(τ)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# SUMMARY
# -----------------------------
print(f"Total pulses emitted: {len(t_emit)}")
print(f"Detector A: {len(detector_A)} detections")
print(f"Detector B: {len(detector_B)} detections")
print(f"Total coincidences recorded: {np.sum(hist)}")
