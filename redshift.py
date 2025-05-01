import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 1.0
T_emit = 1.0
num_pulses = 100
x_comoving = 1.0

# Scale factor (matter-dominated universe): a(t) = t^(2/3)
def a(t):
    return t**(2 / 3)

# Light travels along null geodesic: dx = c / a(t) * dt
def find_arrival_time(t_emit, x_target, dt=0.01):
    t = t_emit
    x = 0.0
    while x < x_target and t < 1e4:  # cap to avoid infinite loops
        x += c / a(t) * dt
        t += dt
    return t

# Emit pulses at proper intervals (start at t=1 to avoid a(0))
t_emit_vals = np.arange(num_pulses) * T_emit + 1.0
t_arrivals = []

for t_emit in t_emit_vals:
    t_arrival = find_arrival_time(t_emit, x_comoving)
    t_arrivals.append(t_arrival)

# Ensure arrays are same length
t_emit_vals = np.array(t_emit_vals[:len(t_arrivals)])
t_arrivals = np.array(t_arrivals)

# Compute pulse-to-pulse intervals
arrival_intervals = np.diff(t_arrivals)

# Compute individual pulse redshifts
pulse_redshifts = arrival_intervals / T_emit - 1
mean_z = np.mean(pulse_redshifts)

# Compute final theoretical redshift from scale factors
z_theoretical = a(t_arrivals[-1]) / a(t_emit_vals[-1]) - 1

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(pulse_redshifts, label="Simulated Redshift per Pulse", color='darkred')
plt.axhline(mean_z, linestyle='--', color='orange', label=f"Mean z = {mean_z:.3f}")
plt.title("Pulse-Based Cosmological Redshift (Matter-Dominated Universe)")
plt.xlabel("Pulse Index")
plt.ylabel("Redshift (z)")
plt.grid(True)
plt.legend()
plt.show()

# Final debug
print(f"Mean simulated redshift: z = {mean_z:.4f}")
print(f"Theoretical redshift (last pulse): z = {z_theoretical:.4f}")
