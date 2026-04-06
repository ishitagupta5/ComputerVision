"""
analyze_depth_data.py — Publication-quality stereo depth accuracy analysis

Generates polished, presentation-ready figures:
  1. Actual vs Reported with confidence band
  2. Mean error by distance (grouped by trial)
  3. Error percentage with accuracy zones
  4. Combined dashboard figure

Usage:
  python analyze_depth_data.py
  python analyze_depth_data.py my_data.csv
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict

# ============================================================
# STYLE SETUP — dark academic theme
# ============================================================
plt.rcParams.update({
    "figure.facecolor": "#0D1117",
    "axes.facecolor": "#161B22",
    "axes.edgecolor": "#30363D",
    "axes.labelcolor": "#E6EDF3",
    "axes.titlepad": 16,
    "text.color": "#E6EDF3",
    "xtick.color": "#8B949E",
    "ytick.color": "#8B949E",
    "grid.color": "#21262D",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#0D1117",
    "savefig.edgecolor": "none",
})

# Accent colors
C_ACCENT = "#58A6FF"
C_GREEN = "#3FB950"
C_ORANGE = "#D29922"
C_RED = "#F85149"
C_PURPLE = "#BC8CFF"
C_CYAN = "#39D2C0"
TRIAL_COLORS = ["#58A6FF", "#3FB950", "#D29922"]
TRIAL_MARKERS = ["o", "s", "D"]

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
            "confidence": float(row["confidence"]),
            "condition": row["condition"],
            "depth_ms": float(row["depth_compute_ms"]),
            "trial": int(row.get("trial", 1)),
        })

if not data:
    print(f"No data in {csv_file}")
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

# Group by distance
dist_abs = defaultdict(list)
dist_pct = defaultdict(list)
for d in data:
    key = round(d["actual_ft"])
    dist_abs[key].append(abs(d["error_ft"]))
    dist_pct[key].append(d["error_pct"])
distances = sorted(dist_abs.keys())

r_squared = np.corrcoef(actual, reported)[0, 1] ** 2
coeffs = np.polyfit(actual, reported, 1)


def subtitle_text(ax, text, y=1.02):
    """Add a muted subtitle above the axes."""
    ax.text(0.0, y, text, transform=ax.transAxes,
            fontsize=9, color="#8B949E", fontstyle="italic")


# ============================================================
# FIGURE 1 — Actual vs Reported
# ============================================================
fig1, ax = plt.subplots(figsize=(8, 8))

max_d = max(max(actual), max(reported)) * 1.15

# ±10 % accuracy band
band_x = np.linspace(0, max_d, 200)
ax.fill_between(band_x, band_x * 0.9, band_x * 1.1,
                color=C_GREEN, alpha=0.07, label="±10 % band")

# Perfect line
ax.plot([0, max_d], [0, max_d], color="#F85149", ls="--", lw=1.8,
        alpha=0.6, label="Perfect accuracy", zorder=2)

# Best-fit
fit_x = np.linspace(0, max_d, 200)
fit_y = np.polyval(coeffs, fit_x)
ax.plot(fit_x, fit_y, color=C_CYAN, lw=2, alpha=0.8,
        label=f"Fit  y = {coeffs[0]:.2f}x {coeffs[1]:+.2f}   R² = {r_squared:.4f}",
        zorder=3)

# Data points by trial
for i, t in enumerate(unique_trials):
    mask = trials == t
    ax.scatter(actual[mask], reported[mask],
               c=TRIAL_COLORS[i], marker=TRIAL_MARKERS[i],
               s=110, alpha=0.9, edgecolors="white", linewidths=0.6,
               label=f"Trial {t}", zorder=4)

ax.set_xlabel("Actual Distance (ft)")
ax.set_ylabel("Reported Distance (ft)")
ax.set_title("Stereo Depth Accuracy", fontweight="bold", fontsize=18)
subtitle_text(ax, "OAK-D Lite  •  SGBM  •  640×480  •  7.49 cm baseline")
ax.set_xlim(0, max_d)
ax.set_ylim(0, max_d)
ax.set_aspect("equal")
ax.legend(loc="upper left", frameon=True, facecolor="#161B22",
          edgecolor="#30363D", framealpha=0.95)
ax.grid(True)

fig1.tight_layout()
fig1.savefig("graph_actual_vs_reported.png")
print("Saved: graph_actual_vs_reported.png")


# ============================================================
# FIGURE 2 — Absolute Error by Distance (grouped bars)
# ============================================================
fig2, ax = plt.subplots(figsize=(10, 5.5))

x = np.arange(len(distances))
w = 0.22

for i, t in enumerate(unique_trials):
    t_data = [d for d in data if d["trial"] == t]
    vals = []
    for dist in distances:
        match = [abs(d["error_ft"]) for d in t_data if round(d["actual_ft"]) == dist]
        vals.append(match[0] if match else 0)
    ax.bar(x + i * w, vals, w, color=TRIAL_COLORS[i], edgecolor="white",
           linewidth=0.4, alpha=0.85, label=f"Trial {t}", zorder=3)

# Mean line
means = [np.mean(dist_abs[d]) for d in distances]
ax.plot(x + w, means, color=C_RED, marker="o", markersize=7,
        linewidth=2, zorder=5, label="Mean", markeredgecolor="white", markeredgewidth=0.8)

for i, m in enumerate(means):
    ax.annotate(f"{m:.2f} ft", (x[i] + w, m), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=9, color=C_RED, fontweight="bold")

ax.set_xticks(x + w)
ax.set_xticklabels([f"{d} ft" for d in distances])
ax.set_xlabel("Actual Distance")
ax.set_ylabel("Absolute Error (ft)")
ax.set_title("Depth Error by Distance", fontweight="bold", fontsize=18)
subtitle_text(ax, f"{num_trials} independent trials  •  {len(data)} total measurements")
ax.legend(frameon=True, facecolor="#161B22", edgecolor="#30363D", framealpha=0.95)
ax.grid(True, axis="y")
ax.set_ylim(bottom=0)

fig2.tight_layout()
fig2.savefig("graph_error_by_distance.png")
print("Saved: graph_error_by_distance.png")


# ============================================================
# FIGURE 3 — Percentage Error with Accuracy Zones
# ============================================================
fig3, ax = plt.subplots(figsize=(10, 5.5))

mean_pcts = [np.mean(dist_pct[d]) for d in distances]
std_pcts = [np.std(dist_pct[d]) for d in distances]

# Accuracy zone backgrounds
ax.axhspan(0, 5, color=C_GREEN, alpha=0.06, zorder=0)
ax.axhspan(5, 10, color=C_ORANGE, alpha=0.06, zorder=0)
ax.axhspan(10, 20, color=C_RED, alpha=0.04, zorder=0)

ax.text(len(distances) - 0.6, 2.5, "EXCELLENT", fontsize=8, color=C_GREEN,
        alpha=0.7, fontweight="bold", ha="right")
ax.text(len(distances) - 0.6, 7.5, "GOOD", fontsize=8, color=C_ORANGE,
        alpha=0.7, fontweight="bold", ha="right")
ax.text(len(distances) - 0.6, 12.5, "FAIR", fontsize=8, color=C_RED,
        alpha=0.5, fontweight="bold", ha="right")

# Threshold lines
ax.axhline(5, color=C_GREEN, ls="--", lw=1, alpha=0.4)
ax.axhline(10, color=C_ORANGE, ls="--", lw=1, alpha=0.4)

# Bar colors based on value
bar_colors = [C_GREEN if m < 5 else C_ORANGE if m < 10 else C_RED for m in mean_pcts]

bars = ax.bar(x, mean_pcts, 0.55, yerr=std_pcts, capsize=5,
              color=bar_colors, edgecolor="white", linewidth=0.5,
              alpha=0.85, zorder=3, error_kw={"ecolor": "#8B949E", "capthick": 1.2})

for i, (m, s) in enumerate(zip(mean_pcts, std_pcts)):
    ax.text(i, m + s + 0.6, f"{m:.1f}%", ha="center", fontsize=10,
            fontweight="bold", color="white", zorder=6)

ax.set_xticks(x)
ax.set_xticklabels([f"{d} ft" for d in distances])
ax.set_xlabel("Actual Distance")
ax.set_ylabel("Mean Error (%)")
ax.set_title("Depth Error Percentage", fontweight="bold", fontsize=18)
subtitle_text(ax, f"Mean across {num_trials} trials  •  Color = accuracy tier")
ax.set_ylim(0, max(mean_pcts) + max(std_pcts) + 3)
ax.grid(True, axis="y")

fig3.tight_layout()
fig3.savefig("graph_error_pct_by_distance.png")
print("Saved: graph_error_pct_by_distance.png")


# ============================================================
# FIGURE 4 — Combined Dashboard (2×2)
# ============================================================
fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
fig4.suptitle("Stereo Depth Accuracy — Full Analysis",
              fontsize=20, fontweight="bold", color="white", y=0.98)
fig4.text(0.5, 0.945,
          "OAK-D Lite  •  SGBM + OpenMP  •  640×480  •  7.49 cm baseline  •  YOLO 11x detection",
          ha="center", fontsize=10, color="#8B949E", fontstyle="italic")

# --- Panel A: Actual vs Reported ---
a = axes[0, 0]
band_x = np.linspace(0, max_d, 200)
a.fill_between(band_x, band_x * 0.9, band_x * 1.1, color=C_GREEN, alpha=0.07)
a.plot([0, max_d], [0, max_d], color=C_RED, ls="--", lw=1.5, alpha=0.5)
a.plot(fit_x, fit_y, color=C_CYAN, lw=2, alpha=0.8)
for i, t in enumerate(unique_trials):
    mask = trials == t
    a.scatter(actual[mask], reported[mask], c=TRIAL_COLORS[i],
              marker=TRIAL_MARKERS[i], s=80, alpha=0.9,
              edgecolors="white", linewidths=0.5, label=f"Trial {t}")
a.set_xlabel("Actual (ft)")
a.set_ylabel("Reported (ft)")
a.set_title("A.  Actual vs Reported", fontweight="bold", fontsize=13, loc="left")
a.set_xlim(0, max_d)
a.set_ylim(0, max_d)
a.set_aspect("equal")
a.legend(fontsize=8, frameon=True, facecolor="#161B22", edgecolor="#30363D")
a.grid(True)
a.text(0.97, 0.05, f"R² = {r_squared:.4f}", transform=a.transAxes,
       ha="right", fontsize=11, color=C_CYAN, fontweight="bold")

# --- Panel B: Grouped Bar Error ---
b = axes[0, 1]
for i, t in enumerate(unique_trials):
    t_data = [d for d in data if d["trial"] == t]
    vals = []
    for dist in distances:
        match = [abs(d["error_ft"]) for d in t_data if round(d["actual_ft"]) == dist]
        vals.append(match[0] if match else 0)
    b.bar(x + i * w, vals, w, color=TRIAL_COLORS[i], edgecolor="white",
          linewidth=0.3, alpha=0.85, label=f"Trial {t}")
b.plot(x + w, means, color=C_RED, marker="o", markersize=5, lw=1.5,
       markeredgecolor="white", markeredgewidth=0.5)
b.set_xticks(x + w)
b.set_xticklabels([f"{d}" for d in distances])
b.set_xlabel("Distance (ft)")
b.set_ylabel("Abs Error (ft)")
b.set_title("B.  Error by Distance", fontweight="bold", fontsize=13, loc="left")
b.legend(fontsize=8, frameon=True, facecolor="#161B22", edgecolor="#30363D")
b.grid(True, axis="y")
b.set_ylim(bottom=0)

# --- Panel C: Percentage Error ---
c = axes[1, 0]
c.axhspan(0, 5, color=C_GREEN, alpha=0.06)
c.axhspan(5, 10, color=C_ORANGE, alpha=0.06)
c.axhspan(10, 20, color=C_RED, alpha=0.04)
c.axhline(5, color=C_GREEN, ls="--", lw=0.8, alpha=0.4)
c.axhline(10, color=C_ORANGE, ls="--", lw=0.8, alpha=0.4)
c.bar(x, mean_pcts, 0.5, yerr=std_pcts, capsize=4,
      color=bar_colors, edgecolor="white", linewidth=0.4, alpha=0.85,
      error_kw={"ecolor": "#8B949E", "capthick": 1})
for i, m in enumerate(mean_pcts):
    c.text(i, m + std_pcts[i] + 0.5, f"{m:.1f}%", ha="center",
           fontsize=9, fontweight="bold", color="white")
c.set_xticks(x)
c.set_xticklabels([f"{d}" for d in distances])
c.set_xlabel("Distance (ft)")
c.set_ylabel("Error (%)")
c.set_title("C.  Error Percentage", fontweight="bold", fontsize=13, loc="left")
c.set_ylim(0, max(mean_pcts) + max(std_pcts) + 3)
c.grid(True, axis="y")

# --- Panel D: Stats Card ---
d_ax = axes[1, 1]
d_ax.axis("off")

stats_text = [
    ("Total Readings", f"{len(data)}"),
    ("Trials", f"{num_trials}"),
    ("Range", f"{min(actual):.0f} – {max(actual):.0f} ft"),
    ("Mean Abs Error", f"{np.mean(abs_errors):.3f} ft"),
    ("Median Abs Error", f"{np.median(abs_errors):.3f} ft"),
    ("Std Deviation", f"{np.std(abs_errors):.3f} ft"),
    ("Mean Error %", f"{np.mean(pct_errors):.1f}%"),
    ("Best Reading", f"{min(abs_errors):.3f} ft  (at {actual[np.argmin(abs_errors)]:.0f} ft)"),
    ("Worst Reading", f"{max(abs_errors):.3f} ft  (at {actual[np.argmax(abs_errors)]:.0f} ft)"),
    ("R²", f"{r_squared:.4f}"),
    ("Avg Compute", f"{np.mean(compute_times):.0f} ms"),
]

d_ax.set_title("D.  Summary Statistics", fontweight="bold", fontsize=13, loc="left")

y_start = 0.92
for i, (label, value) in enumerate(stats_text):
    y = y_start - i * 0.08
    d_ax.text(0.05, y, label, transform=d_ax.transAxes,
              fontsize=11, color="#8B949E", ha="left", va="top")
    d_ax.text(0.95, y, value, transform=d_ax.transAxes,
              fontsize=11, color="white", fontweight="bold", ha="right", va="top",
              fontfamily="monospace")
    if i < len(stats_text) - 1:
        d_ax.axhline(y=y - 0.03, xmin=0.05, xmax=0.95,
                     color="#21262D", lw=0.5, transform=d_ax.transAxes)

fig4.tight_layout(rect=[0, 0, 1, 0.93])
fig4.savefig("graph_dashboard.png")
print("Saved: graph_dashboard.png")


# ============================================================
# FIGURE 5 — Error Trend Line (clean presentation slide)
# ============================================================
fig5, ax = plt.subplots(figsize=(10, 5))

# Individual points
for i, t in enumerate(unique_trials):
    mask = trials == t
    ax.scatter(actual[mask], pct_errors[mask],
               c=TRIAL_COLORS[i], marker=TRIAL_MARKERS[i],
               s=90, alpha=0.7, edgecolors="white", linewidths=0.5,
               label=f"Trial {t}", zorder=4)

# Mean trend line
ax.plot(distances, mean_pcts, color=C_ACCENT, lw=2.5, marker="o",
        markersize=8, markeredgecolor="white", markeredgewidth=1,
        zorder=5, label="Mean trend")

# Fill std band
upper = [m + s for m, s in zip(mean_pcts, std_pcts)]
lower = [max(0, m - s) for m, s in zip(mean_pcts, std_pcts)]
ax.fill_between(distances, lower, upper, color=C_ACCENT, alpha=0.1, zorder=1)

ax.axhline(5, color=C_GREEN, ls="--", lw=1, alpha=0.5, label="5% threshold")
ax.axhline(10, color=C_ORANGE, ls="--", lw=1, alpha=0.5, label="10% threshold")

ax.set_xlabel("Actual Distance (ft)")
ax.set_ylabel("Error (%)")
ax.set_title("Depth Error vs Distance", fontweight="bold", fontsize=18)
subtitle_text(ax, "Error increases with distance as disparity resolution decreases")
ax.legend(frameon=True, facecolor="#161B22", edgecolor="#30363D", framealpha=0.95,
          fontsize=9)
ax.set_ylim(bottom=0)
ax.grid(True)

fig5.tight_layout()
fig5.savefig("graph_error_trend.png")
print("Saved: graph_error_trend.png")


# ============================================================
# TERMINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  SUMMARY STATISTICS")
print("=" * 60)
print(f"  Total readings:        {len(data)}")
print(f"  Trials:                {num_trials}")
print(f"  Distance range:        {min(actual):.0f} – {max(actual):.0f} ft")
print(f"  Mean absolute error:   {np.mean(abs_errors):.3f} ft")
print(f"  Median absolute error: {np.median(abs_errors):.3f} ft")
print(f"  Mean error %:          {np.mean(pct_errors):.1f}%")
print(f"  R²:                    {r_squared:.4f}")
print(f"  Avg compute time:      {np.mean(compute_times):.0f} ms")
print("=" * 60)
print(f"\n  {'Dist':>6s}  {'Mean Err':>9s}  {'Std':>7s}  {'Mean %':>7s}  {'n':>3s}")
print("  " + "-" * 40)
for d in distances:
    e = dist_abs[d]
    p = dist_pct[d]
    print(f"  {d:>4d}ft  {np.mean(e):>8.3f}ft  {np.std(e):>6.3f}  {np.mean(p):>6.1f}%  {len(e):>3d}")
print()

# Save summary
with open("depth_accuracy_summary.txt", "w") as f:
    f.write("STEREO DEPTH ACCURACY — EXPERIMENT SUMMARY\n")
    f.write("=" * 50 + "\n")
    f.write("Camera:       OAK-D Lite\n")
    f.write("Method:       SGBM stereo matching (OpenMP, 8 threads)\n")
    f.write("Detection:    YOLO 11x\n")
    f.write("Resolution:   640×480\n")
    f.write("Focal length: 470.38 px\n")
    f.write("Baseline:     7.49 cm\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total readings:        {len(data)}\n")
    f.write(f"Trials:                {num_trials}\n")
    f.write(f"Distance range:        {min(actual):.0f} – {max(actual):.0f} ft\n")
    f.write(f"Mean absolute error:   {np.mean(abs_errors):.3f} ft\n")
    f.write(f"Median absolute error: {np.median(abs_errors):.3f} ft\n")
    f.write(f"Std deviation:         {np.std(abs_errors):.3f} ft\n")
    f.write(f"Mean error %:          {np.mean(pct_errors):.1f}%\n")
    f.write(f"R²:                    {r_squared:.4f}\n")
    f.write(f"Mean compute time:     {np.mean(compute_times):.0f} ms\n\n")
    f.write(f"{'Dist':>6s}  {'Mean Err':>9s}  {'Std':>7s}  {'Mean %':>7s}  {'n':>3s}\n")
    f.write("-" * 42 + "\n")
    for d in distances:
        e = dist_abs[d]
        p = dist_pct[d]
        f.write(f"{d:>4d}ft  {np.mean(e):>8.3f}ft  {np.std(e):>6.3f}  {np.mean(p):>6.1f}%  {len(e):>3d}\n")

print("Saved: depth_accuracy_summary.txt")
print("All graphs saved!\n")

plt.show()