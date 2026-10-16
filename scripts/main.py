"""
main.py - Run the whole analysis and save the results to the outputs/ folder.

    python scripts/main.py      (from the project folder)

Outputs (saved to the outputs/ folder at the top of the project):
    outputs/thickness_maps.png   raw scan, cleaned scan, detected defects
    outputs/defect_close_up.png  zoom on the most severe defect
    outputs/defects.csv          the defect table (opens in Excel)
    outputs/pipe_3d.html         interactive 3D pipe coloured by thickness
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.graph_objects as go

from simulate import (simulate, NOMINAL_WALL_MM, OUTER_DIAMETER_MM,
                      DEPTH_STEP_M, depth_to_row)
from detect import detect, check_against_truth

# Save to <project>/outputs, wherever the script is run from
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

# A fixed colour scale for every image, so they can be compared directly:
# red = thin (metal loss), white = nominal, blue = thicker (collars).
# The scale is centred on nominal thickness so "white" means "as designed".
VMIN = 3.0
VMAX = 2 * NOMINAL_WALL_MM - VMIN
CMAP = matplotlib.colormaps["RdBu"].copy()
CMAP.set_bad("#9e9e9e")          # missing data shown in grey, never as a colour


def plot_maps(raw, smooth, table, depths_m):
    extent = [0, 360, depths_m[-1], depths_m[0]]   # depth increases downward
    fig, axes = plt.subplots(1, 3, figsize=(15, 9), sharey=True)
    titles = ["1. Raw scan (with noise and missing data)",
              "2. After median filter",
              "3. Detected defects"]
    for ax, data, title in zip(axes, [raw, smooth, smooth], titles):
        im = ax.imshow(data, aspect="auto", cmap=CMAP, vmin=VMIN, vmax=VMAX,
                       extent=extent, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Angle around pipe (deg)")
        ax.set_xticks([0, 90, 180, 270, 360])
    axes[0].set_ylabel("Measured depth (m)")

    # Draw a labelled box around each detected defect
    for num, d in table.iterrows():
        top = d["Depth top (m)"]
        height = d["Length (mm)"] / 1000
        width_deg = d["Width (mm)"] / (np.pi * OUTER_DIAMETER_MM) * 360
        left = d["Angle (deg)"] - width_deg / 2
        axes[2].add_patch(Rectangle((left, top), width_deg, height, fill=False,
                                    edgecolor="black", linewidth=1.2))
        y_text = top + height / 2 + (0.9 if num == 4 else 0)   # avoid overlap with #2
        axes[2].text(left + width_deg + 4, y_text, f"#{num}",
                     fontsize=9, va="center", weight="bold")
    axes[2].text(200, 1231.2, "No data (grey)", fontsize=8, ha="center")

    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label(f"Wall thickness (mm) · nominal {NOMINAL_WALL_MM} mm")
    fig.suptitle("7-inch casing, 1200–1260 m: unrolled ultrasonic thickness map",
                 fontsize=13, weight="bold")
    fig.savefig(f"{OUT}/thickness_maps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_close_up(smooth, table, depths_m):
    """Zoom on the worst defect, with a thickness profile through it."""
    worst = table.iloc[0]
    r = depth_to_row(worst["Depth of deepest point (m)"])
    c = worst["Angle (deg)"]
    r1, r2 = max(r - 60, 0), min(r + 60, len(depths_m) - 1)
    c1, c2 = max(c - 30, 0), min(c + 30, 359)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    im = ax1.imshow(smooth[r1:r2, c1:c2], cmap=CMAP, vmin=VMIN, vmax=VMAX, aspect="auto",
                    extent=[c1, c2, depths_m[r2], depths_m[r1]])
    ax1.set_title(f"Defect #1 close-up: {worst['Type']}, {worst['Wall loss (%)']}% wall loss")
    ax1.set_xlabel("Angle (deg)")
    ax1.set_ylabel("Depth (m)")
    fig.colorbar(im, ax=ax1, label="Wall thickness (mm)")

    ax2.plot(np.arange(c1, c2), smooth[r, c1:c2], color="black")
    ax2.axhline(NOMINAL_WALL_MM, ls="--", color="grey", label=f"Nominal {NOMINAL_WALL_MM} mm")
    ax2.axhline(NOMINAL_WALL_MM * 0.9, ls=":", color="red", label="10% wall-loss limit")
    ax2.set_title(f"Thickness profile around the pipe at {worst['Depth of deepest point (m)']} m")
    ax2.set_xlabel("Angle (deg)")
    ax2.set_ylabel("Wall thickness (mm)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/defect_close_up.png", dpi=130)
    plt.close(fig)


def plot_3d(smooth, depths_m):
    """
    Wrap the flat map back into a pipe. Each reading has a depth (z) and an
    angle (theta); its position in 3D is:
        x = r * cos(theta),  y = r * sin(theta),  z = depth
    The surface colour is the wall thickness. Every 20th row is used to keep
    the file small enough to open quickly.
    """
    step = 20
    theta = np.radians(np.arange(0, 361))                 # 361 closes the ring
    z = depths_m[::step]
    data = smooth[::step]
    data = np.concatenate([data, data[:, :1]], axis=1)    # repeat 0 deg at 360
    r = OUTER_DIAMETER_MM / 2 / 1000                      # radius in metres
    # Exaggerate the radius 20x so the pipe is visible next to 60 m of depth
    R = r * 20
    X = R * np.cos(theta)[None, :] * np.ones((len(z), 1))
    Y = R * np.sin(theta)[None, :] * np.ones((len(z), 1))
    Z = -z[:, None] * np.ones((1, len(theta)))            # depth goes down
    fig = go.Figure(go.Surface(
        x=X, y=Y, z=Z, surfacecolor=np.nan_to_num(data, nan=VMIN),
        colorscale="RdBu", cmin=VMIN, cmax=VMAX,
        colorbar=dict(title="Wall (mm)")))
    fig.update_layout(
        title="3D casing view coloured by wall thickness (radius exaggerated 20x)",
        scene=dict(xaxis_title="", yaxis_title="", zaxis_title="Depth (m)",
                   aspectmode="manual", aspectratio=dict(x=0.4, y=0.4, z=3)),
        margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(f"{OUT}/pipe_3d.html", include_plotlyjs="cdn")


def main():
    raw, truth, depths_m, collar_depths = simulate()
    table, smooth, thin, gaps, collar_rows = detect(raw, depths_m)

    print("\n=== DATA QUALITY ===")
    for top, bottom in gaps:
        print(f"No data from {top:.2f} m to {bottom:.2f} m -> low confidence, no defect calls")
    print(f"Collars recognised as normal geometry: {len(collar_depths)}")

    print("\n=== DEFECTS FOUND (worst first) ===")
    print(table.to_string())
    table.to_csv(f"{OUT}/defects.csv")

    print("\n=== QUALITY CHECK AGAINST PLANTED DEFECTS ===")
    check, false_calls = check_against_truth(table, truth)
    print(check.to_string(index=False))
    print(f"False calls (reported but not real): {false_calls}")

    plot_maps(raw, smooth, table, depths_m)
    plot_close_up(smooth, table, depths_m)
    plot_3d(smooth, depths_m)
    print("\nSaved images, table and 3D view to the 'outputs' folder.")


if __name__ == "__main__":
    main()
