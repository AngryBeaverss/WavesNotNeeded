import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_pulses = 1000
pulse_rate = 10  # average number of pulses per second
quantum_efficiency = 0.6  # probability of detection
detector_dead_time = 0.05  # seconds (no double-hits during this time)
energy_mean = 1.0
energy_std = 0.2

# Generate random pulse emission times (Poisson process)
t_emit = np.cumsum(np.random.exponential(scale=1/pulse_rate, size=num_pulses))
energies = np.random.normal(loc=energy_mean, scale=energy_std, size=num_pulses)

# Detector simulation
t_detections = []
last_detection_time = -np.inf

for t, E in zip(t_emit, energies):
    if t - last_detection_time < detector_dead_time:
        continue  # ignore pulse due to dead time
    p_hit = quantum_efficiency * min(1.0, E / energy_mean)
    if np.random.rand() < p_hit:
        t_detections.append(t)
        last_detection_time = t

t_detections = np.array(t_detections)

# Plot histogram of detection times
plt.figure(figsize=(10, 5))
plt.hist(t_detections, bins=50, color='purple', alpha=0.7, label='Detected Pulses')
plt.title("Photon-Like Pulse Detection Histogram")
plt.xlabel("Time (s)")
plt.ylabel("Counts")
plt.grid(True)
plt.legend()
plt.show()

# Summary
print(f"Total pulses: {num_pulses}")
print(f"Detected pulses: {len(t_detections)}")
print(f"Detection efficiency: {len(t_detections)/num_pulses:.3f}")
