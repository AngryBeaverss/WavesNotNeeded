import pandas as pd
import matplotlib.pyplot as plt

# --- Load the data ---
df = pd.read_csv("forge_visibility_sweep.csv")  # Make sure the CSV is in your working directory

# --- Extract unique parameter values ---
dark_times = sorted(df["dark_time_us"].unique())
angular_spreads = sorted(df["angular_spread_mrad"].unique())

# --- Create plots ---
fig, axs = plt.subplots(len(dark_times), 1, figsize=(10, 4 * len(dark_times)), sharex=True)

for i, dt in enumerate(dark_times):
    ax = axs[i] if len(dark_times) > 1 else axs
    for spread in angular_spreads:
        subset = df[(df["dark_time_us"] == dt) & (df["angular_spread_mrad"] == spread)]
        ax.plot(
            subset["slit_separation_mm"],
            subset["visibility"],
            marker='o',
            label=f"{spread:.1f} mrad"
        )
    ax.set_title(f"Fringe Visibility vs Slit Separation (Dark Time = {dt} µs)")
    ax.set_xlabel("Slit Separation (mm)")
    ax.set_ylabel("Fringe Visibility")
    ax.grid(True)
    ax.legend(title="Angular Spread")

plt.tight_layout()
plt.show()
