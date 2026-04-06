"""
analyze_depth_data.py — Generate accuracy graphs from collected data

Reads depth_accuracy_data.csv (with trial column) and produces:
  1. Scatter plot: actual vs reported distance (color-coded by trial)
  2. Error bar chart: mean absolute error at each distance (across trials)
  3. Error percentage chart: mean % error at each distance
  4. Summary statistics table

Usage:
  python analyze_depth_data.py
  python analyze_depth_data.py my_data.csv
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ============================================================
# LOAD DATA
# ============================================================

csv_file = sys.argv[1] if len(sys.argv) > 1 else "depth_accuracy_data.csv"

data = []
with open(csv_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({
            "actual_ft": float(row["actual_ft"]),
            "reported_ft": float(row["reported_ft"]),
            "error_ft": float(row["error_ft"]),
            "error_pct": float(row["error_pct"]),
            "object": row["object"],
            "confidence": float(row["confidence"]),
            "condition": row["condition"],
            "depth_ms": float(row["depth_compute_ms"]),
            "trial": int(row.get("trial", 1)),
        })

if len(data) == 0:
    print(f"No data found in {csv_file}")
    sys.exit(1)

print(f"Loaded {len(data)} readings from {csv_file}")

actual = np.array([d["actual_ft"] for d in data])
reported = np.array([d["reported_ft"] for d in data])
errors = np.array([d["error_ft"] for d in data])
abs_errors = np.abs(errors)
pct_errors = np.array([d["error_pct"] for d in data])
trials = np.array([d["trial"] for d in data])
compute_times = np.array([d["depth_ms"] for d in data])

unique_trials = sorted(set(trials))
num_trials = len(unique_trials)

# ============================================================
# FIGURE 1: ACTUAL vs REPORTED (color-coded by trial)
# ============================================================
fig1, ax1 = plt.subplots(figsize=(8, 8))

trial_colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
trial_markers = ["o", "s", "D", "^", "v"]

for i, t in enumerate(unique_trials):
    mask = trials == t
    ax1.scatter(actual[mask], reported[mask],
                c=trial_colors[i % len(trial_colors)],
                marker=trial_markers[i % len(trial_markers)],
                s=90, alpha=0.8, edgecolors="black", linewidths=0.5,
                label=f"Trial {t}", zorder=3)

max_dist = max(max(actual), max(reported)) * 1.15
ax1.plot([0, max_dist], [0, max_dist], "r--", linewidth=2, label="Perfect accuracy", zorder=1)

coeffs = np.polyfit(actual, reported, 1)
fit_x = np.linspace(0, max_dist, 100)
fit_y = np.polyval(coeffs, fit_x)
r_squared = np.corrcoef(actual, reported)[0, 1] ** 2
ax1.plot(fit_x, fit_y, "g-", linewidth=1.5, alpha=0.7,
         label=f"Best fit: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}  (R²={r_squared:.4f})", zorder=2)

ax1.set_xlabel("Actual Distance (ft)", fontsize=13)
ax1.set_ylabel("Reported Distance (ft)", fontsize=13)
ax1.set_title("Stereo Depth Accuracy: Actual vs Reported", fontsize=15, fontweight="bold")
ax1.legend(fontsize=10, loc="upper left")
ax1.set_xlim(0, max_dist)
ax1.set_ylim(0, max_dist)
ax1.set_aspect("equal")
ax1.grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig("graph_actual_vs_reported.png", dpi=150)
print("Saved: graph_actual_vs_reported.png")

# ============================================================
# FIGURE 2: ABSOLUTE ERROR BY DISTANCE (mean + std across trials)
# ============================================================

dist_groups = defaultdict(list)
for d in data:
    key = round(d["actual_ft"])
    dist_groups[key].append(abs(d["error_ft"]))

distances_sorted = sorted(dist_groups.keys())
mean_errors = [np.mean(dist_groups[d]) for d in distances_sorted]
std_errors = [np.std(dist_groups[d]) for d in distances_sorted]

fig2, ax2 = plt.subplots(figsize=(10, 6))

bar_colors = ["#4CAF50" if m < 0.5 else "#FF9800" if m < 1.0 else "#F44336" for m in mean_errors]

bars = ax2.bar(range(len(distances_sorted)), mean_errors, yerr=std_errors,
               color=bar_colors, edgecolor="black", capsize=6, alpha=0.85, linewidth=0.8)
ax2.set_xticks(range(len(distances_sorted)))
ax2.set_xticklabels([f"{d} ft" for d in distances_sorted], fontsize=12)
ax2.set_xlabel("Actual Distance", fontsize=13)
ax2.set_ylabel("Mean Absolute Error (ft)", fontsize=13)
ax2.set_title(f"Depth Estimation Error by Distance (n={num_trials} trials)", fontsize=15, fontweight="bold")
ax2.grid(True, axis="y", alpha=0.3)

for i, (mean, std) in enumerate(zip(mean_errors, std_errors)):
    ax2.text(i, mean + std + 0.05, f"{mean:.2f} ft", ha="center", fontsize=10, fontweight="bold")

fig2.tight_layout()
fig2.savefig("graph_error_by_distance.png", dpi=150)
print("Saved: graph_error_by_distance.png")

# ============================================================
# FIGURE 3: PERCENTAGE ERROR BY DISTANCE
# ============================================================

pct_groups = defaultdict(list)
for d in data:
    key = round(d["actual_ft"])
    pct_groups[key].append(d["error_pct"])

mean_pct = [np.mean(pct_groups[d]) for d in distances_sorted]
std_pct = [np.std(pct_groups[d]) for d in distances_sorted]

fig3, ax3 = plt.subplots(figsize=(10, 6))

pct_colors = ["#4CAF50" if m < 5 else "#FF9800" if m < 10 else "#F44336" for m in mean_pct]

ax3.bar(range(len(distances_sorted)), mean_pct, yerr=std_pct,
        color=pct_colors, edgecolor="black", capsize=6, alpha=0.85, linewidth=0.8)
ax3.set_xticks(range(len(distances_sorted)))
ax3.set_xticklabels([f"{d} ft" for d in distances_sorted], fontsize=12)
ax3.set_xlabel("Actual Distance", fontsize=13)
ax3.set_ylabel("Mean Error (%)", fontsize=13)
ax3.set_title(f"Depth Estimation Error % by Distance (n={num_trials} trials)", fontsize=15, fontweight="bold")
ax3.grid(True, axis="y", alpha=0.3)
ax3.axhline(y=10, color="gray", linestyle="--", alpha=0.5, label="10% threshold")
ax3.axhline(y=5, color="green", linestyle="--", alpha=0.4, label="5% threshold")
ax3.legend(fontsize=10)

for i, (mean, std) in enumerate(zip(mean_pct, std_pct)):
    ax3.text(i, mean + std + 0.4, f"{mean:.1f}%", ha="center", fontsize=10, fontweight="bold")

fig3.tight_layout()
fig3.savefig("graph_error_pct_by_distance.png", dpi=150)
print("Saved: graph_error_pct_by_distance.png")

# ============================================================
# FIGURE 4: PER-TRIAL COMPARISON (grouped bars)
# ============================================================

fig4, ax4 = plt.subplots(figsize=(10, 6))

bar_width = 0.25
x_pos = np.arange(len(distances_sorted))

for i, t in enumerate(unique_trials):
    trial_data = [d for d in data if d["trial"] == t]
    trial_errors = []
    for dist in distances_sorted:
        matching = [abs(d["error_ft"]) for d in trial_data if round(d["actual_ft"]) == dist]
        trial_errors.append(matching[0] if matching else 0)
    ax4.bar(x_pos + i * bar_width, trial_errors, bar_width,
            color=trial_colors[i % len(trial_colors)], edgecolor="black",
            linewidth=0.5, alpha=0.85, label=f"Trial {t}")

ax4.set_xticks(x_pos + bar_width * (num_trials - 1) / 2)
ax4.set_xticklabels([f"{d} ft" for d in distances_sorted], fontsize=12)
ax4.set_xlabel("Actual Distance", fontsize=13)
ax4.set_ylabel("Absolute Error (ft)", fontsize=13)
ax4.set_title("Per-Trial Error Comparison", fontsize=15, fontweight="bold")
ax4.legend(fontsize=10)
ax4.grid(True, axis="y", alpha=0.3)

fig4.tight_layout()
fig4.savefig("graph_per_trial_comparison.png", dpi=150)
print("Saved: graph_per_trial_comparison.png")

# ============================================================
# FIGURE 5: COMPUTE TIME
# ============================================================

fig5, ax5 = plt.subplots(figsize=(8, 5))
ax5.scatter(actual, compute_times, c="purple", s=60, alpha=0.6, edgecolors="black", linewidths=0.5)
ax5.axhline(y=np.mean(compute_times), color="red", linestyle="--", alpha=0.5,
            label=f"Mean: {np.mean(compute_times):.0f} ms")
ax5.set_xlabel("Actual Distance (ft)", fontsize=13)
ax5.set_ylabel("Depth Compute Time (ms)", fontsize=13)
ax5.set_title("Compute Time vs Distance", fontsize=15, fontweight="bold")
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

fig5.tight_layout()
fig5.savefig("graph_compute_time.png", dpi=150)
print("Saved: graph_compute_time.png")

# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("  SUMMARY STATISTICS")
print("=" * 60)
print(f"  Total readings:        {len(data)}")
print(f"  Trials:                {num_trials}")
print(f"  Distance range:        {min(actual):.1f} - {max(actual):.1f} ft")
print(f"  Mean absolute error:   {np.mean(abs_errors):.3f} ft")
print(f"  Median absolute error: {np.median(abs_errors):.3f} ft")
print(f"  Std deviation:         {np.std(abs_errors):.3f} ft")
print(f"  Mean error %:          {np.mean(pct_errors):.1f}%")
print(f"  Max error:             {max(abs_errors):.3f} ft at {actual[np.argmax(abs_errors)]:.1f} ft")
print(f"  Best accuracy:         {min(abs_errors):.3f} ft at {actual[np.argmin(abs_errors)]:.1f} ft")
print(f"  R² (fit quality):      {r_squared:.4f}")
print(f"  Mean compute time:     {np.mean(compute_times):.0f} ms")
print("=" * 60)

print("\n  PER-DISTANCE BREAKDOWN:")
print(f"  {'Dist':>6s}  {'Mean Err':>9s}  {'Std':>7s}  {'Mean %':>7s}  {'n':>3s}")
print("  " + "-" * 40)
for d in distances_sorted:
    errs = dist_groups[d]
    pcts = pct_groups[d]
    print(f"  {d:>4d}ft  {np.mean(errs):>8.3f}ft  {np.std(errs):>6.3f}  {np.mean(pcts):>6.1f}%  {len(errs):>3d}")

# Save summary
with open("depth_accuracy_summary.txt", "w") as f:
    f.write("STEREO DEPTH ACCURACY SUMMARY\n")
    f.write(f"{'=' * 50}\n")
    f.write(f"Camera: OAK-D Lite\n")
    f.write(f"Method: SGBM stereo matching (OpenMP parallelized)\n")
    f.write(f"Focal length: 470.38 px\n")
    f.write(f"Baseline: 7.49 cm\n")
    f.write(f"Resolution: 640x480\n")
    f.write(f"{'=' * 50}\n\n")
    f.write(f"Total readings:        {len(data)}\n")
    f.write(f"Trials:                {num_trials}\n")
    f.write(f"Distance range:        {min(actual):.1f} - {max(actual):.1f} ft\n")
    f.write(f"Mean absolute error:   {np.mean(abs_errors):.3f} ft\n")
    f.write(f"Median absolute error: {np.median(abs_errors):.3f} ft\n")
    f.write(f"Std deviation:         {np.std(abs_errors):.3f} ft\n")
    f.write(f"Mean error %:          {np.mean(pct_errors):.1f}%\n")
    f.write(f"R²:                    {r_squared:.4f}\n")
    f.write(f"Mean compute time:     {np.mean(compute_times):.0f} ms\n")
    f.write(f"\nPER-DISTANCE BREAKDOWN:\n")
    f.write(f"{'Dist':>6s}  {'Mean Err':>9s}  {'Std':>7s}  {'Mean %':>7s}  {'n':>3s}\n")
    f.write("-" * 40 + "\n")
    for d in distances_sorted:
        errs = dist_groups[d]
        pcts = pct_groups[d]
        f.write(f"{d:>4d}ft  {np.mean(errs):>8.3f}ft  {np.std(errs):>6.3f}  {np.mean(pcts):>6.1f}%  {len(errs):>3d}\n")

print("\nSaved: depth_accuracy_summary.txt")
print("\nAll graphs saved! Open the PNG files to view them.")

plt.show()