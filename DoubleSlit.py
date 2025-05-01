import numpy as np
import matplotlib.pyplot as plt

# --- Realistic Detector & Pulse Parameters ---
num_pulses = 100_000
pulse_rate = 10_000                  # Hz
quantum_efficiency = 0.7
detector_dead_time = 100e-9          # 100 ns
dark_rate = 10                       # false counts/sec
jitter_std = 0.5e-9
a = 5e-6  # 5 µm slit width# seconds

# --- Optical Configuration for High Visibility ---
lambda_eff = 800e-9                  # 800 nm wavelength (IR)
d = 20e-6                            # 20 μm slit separation
L = 1.0                              # distance to screen (m)
screen_width = 0.04                  # 1 cm total width
num_bins = 600                       # resolution of screen

# --- Precompute pulse emission times ---
t_emit = np.cumsum(np.random.exponential(scale=1 / pulse_rate, size=num_pulses))
y_screen = np.linspace(-screen_width / 2, screen_width / 2, num_bins)
detections = np.zeros(num_bins)
last_detection_time = -np.inf

# --- Main Detection Loop ---
for t in t_emit:
    if t - last_detection_time < detector_dead_time:
        continue

    t += np.random.normal(0, jitter_std)

    r1 = np.sqrt(L ** 2 + (y_screen + d / 2) ** 2)
    r2 = np.sqrt(L ** 2 + (y_screen - d / 2) ** 2)
    delta_r = r1 - r2

    # Interference term
    interference = np.cos(np.pi * delta_r / lambda_eff) ** 2

    # Diffraction envelope
    sinc_arg = np.pi * a * y_screen / (lambda_eff * L)
    envelope = (np.sinc(sinc_arg / np.pi)) ** 2  # normalize sinc argument

    # Combined probability with diffraction
    probabilities = quantum_efficiency * interference * envelope
    probabilities /= np.sum(probabilities)

    chosen_bin = np.random.choice(np.arange(num_bins), p=probabilities)
    detections[chosen_bin] += 1
    last_detection_time = t

# --- Simulate Dark Counts ---
total_time = t_emit[-1]
expected_dark_counts = np.random.poisson(dark_rate * total_time)
dark_positions = np.random.choice(np.arange(num_bins), expected_dark_counts)
for idx in dark_positions:
    detections[idx] += 1

# --- Plot the Final Interference Pattern ---
plt.figure(figsize=(10, 5))
plt.bar(y_screen, detections, width=screen_width / num_bins, color='royalblue', alpha=0.85)
plt.title("Pulse-Based Double-Slit Interference with Realistic Detection")
plt.xlabel("Screen Position (m)")
plt.ylabel("Photon Counts")
plt.grid(True)
plt.tight_layout()
plt.show()
