import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_pulses = 1000
pulse_rate = 20  # avg pulses/sec
quantum_efficiency = 0.7
dead_time = 0.05
window_size = 1.0  # seconds to compute g2(τ)
bin_width = 0.01   # τ resolution

# Generate random pulse stream (Poisson process)
t_emit = np.cumsum(np.random.exponential(scale=1/pulse_rate, size=num_pulses))

# Simulate two detectors with independent response
def detect_pulses(t_emit, eta, dead):
    detections = []
    last_t = -np.inf
    for t in t_emit:
        if t - last_t < dead:
            continue
        if np.random.rand() < eta:
            detections.append(t)
            last_t = t
    return np.array(detections)

detector_A = detect_pulses(t_emit, quantum_efficiency, dead_time)
detector_B = detect_pulses(t_emit, quantum_efficiency, dead_time)

# Build τ histogram of coincidences
taus = []
for t1 in detector_A:
    # Include all B times within window around t1
    mask = np.abs(detector_B - t1) <= window_size
    taus.extend(detector_B[mask] - t1)

taus = np.array(taus)
bins = np.arange(-window_size, window_size + bin_width, bin_width)
hist, edges = np.histogram(taus, bins=bins)
bin_centers = (edges[:-1] + edges[1:]) / 2

# Normalize to get g²(τ)
g2_tau = hist / np.mean(hist)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(bin_centers, g2_tau, drawstyle='steps-mid', color='navy')
plt.axhline(1.0, color='gray', linestyle='--', label="Poissonian baseline (g²=1)")
plt.title("Second-Order Correlation Function g²(τ) from Pulse Detections")
plt.xlabel("Delay τ (seconds)")
plt.ylabel("g²(τ)")
plt.grid(True)
plt.legend()
plt.show()

# Debug print
print(f"Total coincidences measured: {np.sum(hist)}")
