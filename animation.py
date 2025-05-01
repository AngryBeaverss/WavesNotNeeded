import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Realistic Detector & Pulse Parameters ---
num_pulses = 100_000
pulse_rate = 10_000                  # Hz
quantum_efficiency = 0.7
detector_dead_time = 100e-9          # 100 ns
dark_rate = 10                       # false counts/sec
jitter_std = 0.5e-9                  # seconds
a = 5e-6                              # 5 µm slit width

# --- Optical Configuration for High Visibility ---
lambda_eff = 800e-9                  # 800 nm wavelength (IR)
d = 20e-6                            # 20 μm slit separation
L = 1.0                              # distance to screen (m)
screen_width = 0.04                  # 4 cm total width
num_bins = 600                       # resolution of screen

# --- Precompute pulse emission times ---
t_emit = np.cumsum(np.random.exponential(scale=1 / pulse_rate, size=num_pulses))
y_screen = np.linspace(-screen_width / 2, screen_width / 2, num_bins)
detections = np.zeros(num_bins)
last_detection_time = [-np.inf]  # Use a mutable list for animation context

# --- Plot Setup ---
fig, ax = plt.subplots(figsize=(10, 5))
bar_container = ax.bar(y_screen, np.zeros_like(y_screen), width=screen_width / num_bins, color='royalblue', alpha=0.85)
ax.set_title("Photon-by-Photon Double-Slit Interference Build-up")
ax.set_xlabel("Screen Position (m)")
ax.set_ylabel("Photon Counts")
ax.set_ylim(0, 200)  # You can adjust this based on how tall you want the bars
ax.grid(True)

# Animation: update every few detections
frames = 500  # total animation frames
step = len(t_emit) // frames  # how many detections per frame
hist = np.zeros_like(detections)

def update(frame):
    start = frame * step
    end = min((frame + 1) * step, len(t_emit))

    for i in range(start, end):
        t = t_emit[i]
        if t - last_detection_time[0] < detector_dead_time:
            continue

        t += np.random.normal(0, jitter_std)

        # Geometric paths from both slits
        r1 = np.sqrt(L**2 + (y_screen + d / 2)**2)
        r2 = np.sqrt(L**2 + (y_screen - d / 2)**2)
        delta_r = r1 - r2

        # Interference + envelope
        interference = np.cos(np.pi * delta_r / lambda_eff) ** 2
        sinc_arg = np.pi * a * y_screen / (lambda_eff * L)
        envelope = (np.sinc(sinc_arg / np.pi)) ** 2
        probabilities = quantum_efficiency * interference * envelope
        probabilities /= np.sum(probabilities)

        # Random photon detection
        chosen_bin = np.random.choice(np.arange(num_bins), p=probabilities)
        hist[chosen_bin] += 1
        last_detection_time[0] = t

    for rect, h in zip(bar_container, hist):
        rect.set_height(h)

    return bar_container

ani = animation.FuncAnimation(fig, update, frames=frames, blit=False, repeat=False)
plt.tight_layout()
plt.show()
