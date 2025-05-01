import numpy as np
import matplotlib.pyplot as plt

# Parameters
c = 1.0
T_emit = 1.0  # proper interval in emitter frame
v = 0.5 * c   # emitter velocity
num_pulses = 100

# Lorentz factor
gamma = 1 / np.sqrt(1 - (v / c)**2)

# Time of arrival for each pulse
# Using: t_arrival = γ * τ * (1 + v/c)
tau = np.arange(num_pulses) * T_emit  # emitter's proper time
t_arrival = gamma * tau * (1 + v / c)

# Calculate intervals
arrival_intervals = np.diff(t_arrival)

# Expected from relativistic Doppler
doppler_factor = np.sqrt((1 + v/c) / (1 - v/c))
expected_interval = T_emit * doppler_factor

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(arrival_intervals, label="Simulated Intervals (Relativistic)", color='blue')
plt.axhline(expected_interval, color='green', linestyle='--', label=f"Expected Doppler Interval = {expected_interval:.3f}")
plt.title("Relativistic Pulse-Based Redshift")
plt.xlabel("Pulse Index")
plt.ylabel("Interval Between Arrivals")
plt.grid(True)
plt.legend()
plt.show()

# Debug print
print("γ =", gamma)
print("Expected Doppler Interval:", expected_interval)
print("Mean Simulated Interval:", np.mean(arrival_intervals))
