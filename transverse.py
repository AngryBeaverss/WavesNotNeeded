# Modified: Transverse Doppler with constant radial distance
# Emitter moves along x-axis at fixed y offset from observer

import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 1.0
T_emit = 1.0
v = 0.6 * c
num_pulses = 100
y_fixed = 50.0  # constant distance from observer along y-axis

gamma = 1 / np.sqrt(1 - (v / c)**2)
tau = np.arange(num_pulses) * T_emit

# Emission events in lab frame
t_emit = gamma * tau
x_emit = gamma * v * tau

# Travel time: assume observer at origin (0, 0)
distance = np.full(num_pulses, y_fixed)  # fixed distance (transverse path)
travel_times = distance / c
t_arrival = t_emit + travel_times
arrival_intervals = np.diff(t_arrival)

# Theoretical transverse Doppler interval
expected_interval = gamma * T_emit

# Plot
plt.figure(figsize=(10, 6))
plt.plot(arrival_intervals, label="Simulated Arrival Intervals", color="blue")
plt.axhline(expected_interval, color="green", linestyle="--",
            label=f"Expected Transverse Interval = {expected_interval:.3f}")
plt.title("Transverse Doppler Effect (Pure Geometry)")
plt.xlabel("Pulse Index")
plt.ylabel("Interval Between Arrivals")
plt.legend()
plt.grid(True)
plt.show()

print(f"Lorentz gamma: {gamma:.5f}")
print(f"Expected Interval: {expected_interval:.5f}")
print(f"Simulated Mean Interval: {np.mean(arrival_intervals):.5f}")
