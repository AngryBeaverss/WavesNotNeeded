import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

# Constants
h = 6.62607015e-34  # Planck (J·s)
c = 299792458       # speed of light (m/s)
eV = 1.60218e-19    # 1 eV in J

def energy_to_wavelength_nm(energy_ev):
    """Convert energy in eV to wavelength in nm"""
    return (h * c / (energy_ev * eV)) * 1e9

def wavelength_to_color(wavelength_nm):
    """Return color based on wavelength category"""
    if wavelength_nm < 400:
        return "purple", "UV"
    elif 400 <= wavelength_nm <= 700:
        return "red", "Visible"
    else:
        return "brown", "IR"

def simulate_transitions(n_max=10, trials=20000):
    transitions = {}
    for _ in range(trials):
        n = random.randint(2, n_max)
        m_choices = [i for i in range(1, n)]
        m = random.choice(m_choices)
        delta_l = abs((n - 1) - (m - 1))
        if delta_l == 1:
            E_n = -13.6 / n**2
            E_m = -13.6 / m**2
            delta_E = abs(E_n - E_m)
            label = f"{n} → {m}"
            transitions.setdefault(label, []).append(delta_E)
    return transitions

def plot_transitions(transitions):
    # Flatten all energies for histogram
    all_energies = [e for lst in transitions.values() for e in lst]

    plt.figure(figsize=(10, 6))
    plt.hist(all_energies, bins=150, color="black", alpha=0.85)
    plt.title("Forge Emission Spectrum with Δℓ = ±1 Selection Rule")
    plt.xlabel("Transition Energy (eV)")
    plt.ylabel("Counts")
    plt.grid(True)

    legend_patches = {}
    for label, energies in transitions.items():
        if not energies:
            continue
        avg_E = np.mean(energies)
        wl = energy_to_wavelength_nm(avg_E)
        color, category = wavelength_to_color(wl)
        plt.axvline(x=avg_E, color=color, linestyle='--', linewidth=1.5)
        plt.text(avg_E + 0.05, plt.ylim()[1] * 0.7, f"{label} ({category})",
                 rotation=90, verticalalignment='center', color=color, fontsize=9)
        legend_patches[category] = mpatches.Patch(color=color, label=category)

    # Show legend
    plt.legend(handles=legend_patches.values(), loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.show()

# Run everything
transitions = simulate_transitions(n_max=10, trials=20000)
plot_transitions(transitions)

def simulate_absorption_spectrum(n_max=10, trials=20000, energy_resolution=0.01):
    allowed_transitions = []
    for _ in range(trials):
        m = random.randint(1, n_max - 1)
        n_choices = [i for i in range(m + 1, n_max + 1)]
        n = random.choice(n_choices)
        delta_l = abs((n - 1) - (m - 1))
        if delta_l == 1:
            E_m = -13.6 / m**2
            E_n = -13.6 / n**2
            delta_E = abs(E_n - E_m)
            allowed_transitions.append(delta_E)
    return allowed_transitions

# Simulate absorption under Forge constraints
absorbed_energies = simulate_absorption_spectrum()

# Plot the absorption spectrum
plt.figure(figsize=(10, 6))
plt.hist(absorbed_energies, bins=150, color="blue", alpha=0.85)
plt.title("Forge Absorption Spectrum with Δℓ = ±1 Selection Rule")
plt.xlabel("Absorbed Energy (eV)")
plt.ylabel("Counts")
plt.grid(True)
plt.tight_layout()
plt.show()