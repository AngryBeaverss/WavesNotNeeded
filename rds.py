import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 3e8  # Speed of light in m/s
T_rest = 1e-9  # Pulse interval in emitter's rest frame (1 ns)
v = 0.8 * c  # Relative velocity between emitter and observer
gamma = 1 / np.sqrt(1 - (v / c)**2)  # Lorentz factor
theta = 0  # Angle between motion and line of sight (0 = head-on)

# Number of pulses
num_pulses = 100

# Emission times in emitter's frame
emission_times = np.arange(0, num_pulses * T_rest, T_rest)

# Observed time interval using pulse Doppler shift law
T_obs = gamma * (1 + (v / c) * np.cos(theta)) * T_rest
arrival_times = np.arange(0, num_pulses * T_obs, T_obs)

# Define a baseline distance between emitter and observer
d = 10  # meters

# Compute relativistically corrected pulse arrival times
# Travel time adjusted by relativistic factor
travel_time = d / (c * gamma * (1 + (v / c) * np.cos(theta)))

# Adjusted arrival times including realistic delay
arrival_times_with_delay = emission_times + travel_time + (np.arange(num_pulses) * (T_obs - T_rest))

# Plot the updated results
plt.figure(figsize=(10, 5))
plt.plot(emission_times * 1e9, np.ones_like(emission_times), 'o', label='Emission (ns)', alpha=0.6)
plt.plot(arrival_times_with_delay * 1e9, np.ones_like(arrival_times_with_delay) + 0.1, 'x', label='Observation (ns)', alpha=0.6)
plt.xlabel("Time (nanoseconds)")
plt.yticks([])
plt.title("Relativistic Doppler Shift with Light Travel Delay")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
