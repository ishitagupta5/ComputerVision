"""
analyze_depth_data.py — Generate accuracy graphs from collected data

Reads depth_accuracy_data.csv and produces:
  1. Scatter plot: actual vs reported distance (with perfect accuracy line)
  2. Error bar chart: absolute error at each distance
  3. Error percentage chart: % error at each distance
  4. Condition comparison: accuracy across lighting conditions
  5. Summary statistics table

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
conditions = [d["condition"] for d in data]

# ============================================================
# FIGURE 1: ACTUAL vs REPORTED (scatter + perfect line)
# ============================================================
fig1, ax1 = plt.subplots(figsize=(8, 8))

ax1.scatter(actual, reported, c="steelblue", s=80, alpha=0.7, edgecolors="navy", label="Measurements")

# Perfect accuracy line
max_dist = max(max(actual), max(reported)) * 1.1
ax1.plot([0, max_dist], [0, max_dist], "r--", linewidth=2, label="Perfect accuracy")

# Best fit line
coeffs = np.polyfit(actual, reported, 1)
fit_x = np.linspace(0, max_dist, 100)
fit_y = np.polyval(coeffs, fit_x)
ax1.plot(fit_x, fit_y, "g-", linewidth=1.5, alpha=0.7,
         label=f"Best fit: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}")

ax1.set_xlabel("Actual Distance (ft)", fontsize=13)
ax1.set_ylabel("Reported Distance (ft)", fontsize=13)
ax1.set_title("Stereo Depth Accuracy: Actual vs Reported", fontsize=15, fontweight="bold")
ax1.legend(fontsize=11)
ax1.set_xlim(0, max_dist)
ax1.set_ylim(0, max_dist)
ax1.set_aspect("equal")
ax1.grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig("graph_actual_vs_reported.png", dpi=150)
print("Saved: graph_actual_vs_reported.png")

# ============================================================
# FIGURE 2: ABSOLUTE ERROR BY DISTANCE
# ============================================================

# Group by actual distance
dist_groups = defaultdict(list)
for d in data:
    key = round(d["actual_ft"])
    dist_groups[key].append(abs(d["error_ft"]))

distances_sorted = sorted(dist_groups.keys())
mean_errors = [np.mean(dist_groups[d]) for d in distances_sorted]
std_errors = [np.std(dist_groups[d]) for d in distances_sorted]

fig2, ax2 = plt.subplots(figsize=(10, 6))

bars = ax2.bar(range(len(distances_sorted)), mean_errors, yerr=std_errors,
               color="steelblue", edgecolor="navy", capsize=5, alpha=0.8)
ax2.set_xticks(range(len(distances_sorted)))
ax2.set_xticklabels([f"{d} ft" for d in distances_sorted], fontsize=11)
ax2.set_xlabel("Actual Distance", fontsize=13)
ax2.set_ylabel("Absolute Error (ft)", fontsize=13)
ax2.set_title("Depth Estimation Error by Distance", fontsize=15, fontweight="bold")
ax2.grid(True, axis="y", alpha=0.3)

for i, (mean, std) in enumerate(zip(mean_errors, std_errors)):
    ax2.text(i, mean + std + 0.1, f"{mean:.2f}", ha="center", fontsize=9, fontweight="bold")

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

ax3.bar(range(len(distances_sorted)), mean_pct, yerr=std_pct,
        color="coral", edgecolor="darkred", capsize=5, alpha=0.8)
ax3.set_xticks(range(len(distances_sorted)))
ax3.set_xticklabels([f"{d} ft" for d in distances_sorted], fontsize=11)
ax3.set_xlabel("Actual Distance", fontsize=13)
ax3.set_ylabel("Error (%)", fontsize=13)
ax3.set_title("Depth Estimation Error % by Distance", fontsize=15, fontweight="bold")
ax3.grid(True, axis="y", alpha=0.3)
ax3.axhline(y=10, color="green", linestyle="--", alpha=0.5, label="10% threshold")
ax3.legend(fontsize=11)

for i, (mean, std) in enumerate(zip(mean_pct, std_pct)):
    ax3.text(i, mean + std + 0.5, f"{mean:.1f}%", ha="center", fontsize=9, fontweight="bold")

fig3.tight_layout()
fig3.savefig("graph_error_pct_by_distance.png", dpi=150)
print("Saved: graph_error_pct_by_distance.png")

# ============================================================
# FIGURE 4: ACCURACY BY LIGHTING CONDITION
# ============================================================

unique_conditions = list(set(conditions))
if len(unique_conditions) > 1:
    cond_errors = defaultdict(list)
    for d in data:
        cond_errors[d["condition"]].append(abs(d["error_ft"]))

    cond_sorted = sorted(cond_errors.keys())
    cond_means = [np.mean(cond_errors[c]) for c in cond_sorted]
    cond_stds = [np.std(cond_errors[c]) for c in cond_sorted]
    cond_counts = [len(cond_errors[c]) for c in cond_sorted]

    fig4, ax4 = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(cond_sorted)))
    bars = ax4.bar(range(len(cond_sorted)), cond_means, yerr=cond_stds,
                   color=colors, edgecolor="gray", capsize=5, alpha=0.8)
    ax4.set_xticks(range(len(cond_sorted)))
    ax4.set_xticklabels([c.replace("_", "\n") for c in cond_sorted], fontsize=10)
    ax4.set_xlabel("Lighting Condition", fontsize=13)
    ax4.set_ylabel("Mean Absolute Error (ft)", fontsize=13)
    ax4.set_title("Depth Accuracy by Lighting Condition", fontsize=15, fontweight="bold")
    ax4.grid(True, axis="y", alpha=0.3)

    for i, (mean, n) in enumerate(zip(cond_means, cond_counts)):
        ax4.text(i, mean + cond_stds[i] + 0.1, f"{mean:.2f}ft\n(n={n})",
                 ha="center", fontsize=9)

    fig4.tight_layout()
    fig4.savefig("graph_accuracy_by_condition.png", dpi=150)
    print("Saved: graph_accuracy_by_condition.png")
else:
    print("Only one condition found — skipping condition comparison graph")

# ============================================================
# FIGURE 5: COMPUTE TIME vs DISTANCE
# ============================================================

fig5, ax5 = plt.subplots(figsize=(8, 5))
compute_times = [d["depth_ms"] for d in data]
ax5.scatter(actual, compute_times, c="purple", s=60, alpha=0.6)
ax5.set_xlabel("Actual Distance (ft)", fontsize=13)
ax5.set_ylabel("Depth Compute Time (ms)", fontsize=13)
ax5.set_title("Compute Time vs Distance", fontsize=15, fontweight="bold")
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
print(f"  Distance range:        {min(actual):.1f} - {max(actual):.1f} ft")
print(f"  Mean absolute error:   {np.mean(abs_errors):.3f} ft")
print(f"  Median absolute error: {np.median(abs_errors):.3f} ft")
print(f"  Std deviation:         {np.std(abs_errors):.3f} ft")
print(f"  Mean error %:          {np.mean(pct_errors):.1f}%")
print(f"  Max error:             {max(abs_errors):.3f} ft at {actual[np.argmax(abs_errors)]:.1f} ft")
print(f"  Best accuracy:         {min(abs_errors):.3f} ft at {actual[np.argmin(abs_errors)]:.1f} ft")
print(f"  R² (fit quality):      {np.corrcoef(actual, reported)[0,1]**2:.4f}")
print(f"  Mean compute time:     {np.mean(compute_times):.1f} ms")
print(f"  Conditions tested:     {', '.join(unique_conditions)}")
print("=" * 60)

# Save summary to text file
with open("depth_accuracy_summary.txt", "w") as f:
    f.write("STEREO DEPTH ACCURACY SUMMARY\n")
    f.write("=" * 40 + "\n")
    f.write(f"Total readings:        {len(data)}\n")
    f.write(f"Distance range:        {min(actual):.1f} - {max(actual):.1f} ft\n")
    f.write(f"Mean absolute error:   {np.mean(abs_errors):.3f} ft\n")
    f.write(f"Median absolute error: {np.median(abs_errors):.3f} ft\n")
    f.write(f"Std deviation:         {np.std(abs_errors):.3f} ft\n")
    f.write(f"Mean error %:          {np.mean(pct_errors):.1f}%\n")
    f.write(f"R²:                    {np.corrcoef(actual, reported)[0,1]**2:.4f}\n")
    f.write(f"Mean compute time:     {np.mean(compute_times):.1f} ms\n")
    f.write(f"Conditions:            {', '.join(unique_conditions)}\n")
    f.write("\nPER-DISTANCE BREAKDOWN:\n")
    for d in distances_sorted:
        errs = dist_groups[d]
        pcts = pct_groups[d]
        f.write(f"  {d:>3d} ft: mean_err={np.mean(errs):.3f}ft "
                f"std={np.std(errs):.3f}ft "
                f"pct={np.mean(pcts):.1f}% "
                f"n={len(errs)}\n")

print("\nSaved: depth_accuracy_summary.txt")
print("\nAll graphs saved! Open the PNG files to view them.")

plt.show()