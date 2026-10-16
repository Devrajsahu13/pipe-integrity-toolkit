"""
detect.py - Find, measure and classify wall-loss defects in a thickness map.

The workflow is the same one an analyst follows:
  1. Check data quality       -> where is data missing?
  2. Clean the data           -> median filter removes speckle noise
  3. Recognize normal geometry -> collars are thicker, not defects
  4. Find thin areas          -> threshold at a % wall-loss limit
  5. Group and measure        -> each defect's location, size, depth
  6. Classify and rank        -> pit / general corrosion / groove, severity
"""

import numpy as np
import pandas as pd
from scipy import ndimage

from simulate import (NOMINAL_WALL_MM, OUTER_DIAMETER_MM, DEPTH_STEP_M,
                      ANGLE_STEP_DEG)

WALL_LOSS_LIMIT_PCT = 10    # report anything thinner than 90% of nominal
MIN_DEFECT_PIXELS = 20      # ignore tiny specks left after filtering


def find_missing_data(thk, depths_m):
    """Step 1: list depth intervals where readings are missing (NaN)."""
    missing_rows = np.isnan(thk).any(axis=1)
    labels, n = ndimage.label(missing_rows)
    gaps = []
    for i in range(1, n + 1):
        rows = np.where(labels == i)[0]
        gaps.append((depths_m[rows[0]], depths_m[rows[-1]]))
    return gaps


def clean(thk):
    """
    Step 2: remove speckle noise with a median filter.
    Why median? It replaces each reading with the middle value of its
    neighbours, so isolated spikes vanish but the edges of real defects stay
    sharp. An average (mean) filter would blur defect edges and make pits
    look shallower than they are.
    Missing readings are temporarily filled with nominal thickness so the
    filter can run; they are put back as NaN afterwards.
    """
    missing = np.isnan(thk)
    filled = np.where(missing, NOMINAL_WALL_MM, thk)
    smooth = ndimage.median_filter(filled, size=(5, 5), mode="nearest")
    smooth[missing] = np.nan
    return smooth


def find_collars(smooth):
    """
    Step 3: collars show as a full ring of extra thickness. We find rows whose
    average thickness is well above nominal and exclude them from defect search.
    """
    row_mean = np.nanmean(smooth, axis=1)
    collar_rows = row_mean > NOMINAL_WALL_MM + 1.5
    # widen slightly so the collar edges are excluded too
    return ndimage.binary_dilation(collar_rows, iterations=5)


def classify(length_m, width_deg):
    """Step 6a: simple shape rules, like an analyst's first judgement."""
    if length_m >= 1.0 and width_deg <= 25:          # long and narrow
        return "Wear groove (axial)"
    if length_m >= 1.0 or width_deg >= 30:
        return "General corrosion"
    return "Pitting"


def severity(loss_pct):
    """Step 6b: example severity bands (real ones come from client criteria)."""
    if loss_pct >= 60:
        return "Severe"
    if loss_pct >= 40:
        return "High"
    if loss_pct >= 20:
        return "Moderate"
    return "Low"


def angle_to_clock(angle_deg):
    """0 deg = 12 o'clock, each clock hour = 30 deg."""
    hour = round(angle_deg / 30) % 12
    return f"{12 if hour == 0 else hour} o'clock"


def detect(thk, depths_m):
    """Run the full workflow. Returns (defects table, cleaned map, mask, gaps)."""
    gaps = find_missing_data(thk, depths_m)
    smooth = clean(thk)
    collar_rows = find_collars(smooth)

    # Step 4: threshold - flag readings with more than the allowed wall loss
    limit_mm = NOMINAL_WALL_MM * (1 - WALL_LOSS_LIMIT_PCT / 100)
    thin = np.nan_to_num(smooth, nan=NOMINAL_WALL_MM) < limit_mm
    thin[collar_rows, :] = False

    # Step 5a: join pieces of the same defect. Noise can break one shallow
    # defect (like a long groove) into separate pieces, so we close small gaps
    # along the pipe before grouping. The joined mask is only used to group.
    joined = ndimage.binary_closing(thin, structure=np.ones((41, 3)))
    joined[collar_rows, :] = False

    # Step 5b: measure depth on LIGHTLY filtered data (3x3). The 5x5 filter is
    # good for finding defects, but it shrinks small pits and makes them look
    # shallower - a dangerous under-call. Analysts confirm the deepest point
    # on data that keeps more detail.
    missing = np.isnan(thk)
    light = ndimage.median_filter(np.where(missing, NOMINAL_WALL_MM, thk),
                                  size=(3, 3), mode="nearest")
    light[missing] = np.nan

    # Step 5c: group touching thin readings into individual defects
    labels, n = ndimage.label(joined)
    circumference_mm = np.pi * OUTER_DIAMETER_MM
    rows_out = []
    for i in range(1, n + 1):
        region = labels == i
        if region.sum() < MIN_DEFECT_PIXELS:
            continue                                  # leftover noise
        r_idx, c_idx = np.where(region & thin)        # size from real thin readings
        if len(r_idx) < MIN_DEFECT_PIXELS:
            continue
        # deepest point, measured on the lightly filtered data
        deepest = np.nanargmin(np.where(region, light, np.inf))
        dr, dc = np.unravel_index(deepest, light.shape)
        min_thk = light[dr, dc]
        loss_pct = (NOMINAL_WALL_MM - min_thk) / NOMINAL_WALL_MM * 100
        length_m = (r_idx.max() - r_idx.min() + 1) * DEPTH_STEP_M
        width_deg = (c_idx.max() - c_idx.min() + 1) * ANGLE_STEP_DEG
        rows_out.append({
            "Depth top (m)": round(depths_m[r_idx.min()], 2),
            "Depth of deepest point (m)": round(depths_m[dr], 2),
            "Angle (deg)": int(dc * ANGLE_STEP_DEG),
            "Clock position": angle_to_clock(dc * ANGLE_STEP_DEG),
            "Length (mm)": round(length_m * 1000),
            "Width (mm)": round(width_deg / 360 * circumference_mm),
            "Min wall (mm)": round(min_thk, 2),
            "Wall loss (%)": round(loss_pct, 1),
            "Type": classify(length_m, width_deg),
            "Severity": severity(loss_pct),
        })

    table = pd.DataFrame(rows_out)
    if not table.empty:
        table = table.sort_values("Wall loss (%)", ascending=False).reset_index(drop=True)
        table.index = table.index + 1
        table.index.name = "Defect #"
    return table, smooth, thin, gaps, collar_rows


def check_against_truth(table, truth):
    """
    Quality check on our own method: did we find every planted defect,
    and did we report anything that isn't real (a false call)?
    A detection matches a planted defect if their depth ranges overlap
    (with 0.5 m tolerance) and the angle is within 15 deg. One planted defect
    can match several pieces - e.g. a groove split where it crosses a collar.
    """
    top = table["Depth top (m)"]
    bottom = top + table["Length (mm)"] / 1000
    found, matched = [], set()
    for t in truth:
        t_top = t["depth_m"] - t["length_m"] / 2
        t_bottom = t["depth_m"] + t["length_m"] / 2
        m = table[(t_top <= bottom + 0.5) & (t_bottom >= top - 0.5) &
                  (abs(table["Angle (deg)"] - t["angle_deg"]) < 15)]
        matched.update(m.index)
        found.append({"Planted defect": t["type"], "Depth (m)": t["depth_m"],
                      "Angle (deg)": t["angle_deg"],
                      "Found?": ("Yes (" + ", ".join(f"#{i}" for i in m.index) + ")") if len(m) else "MISSED",
                      "Reported wall loss (%)": m["Wall loss (%)"].max() if len(m) else None,
                      "Planted wall loss (%)": t["loss_pct"]})
    false_calls = len(table) - len(matched)
    return pd.DataFrame(found), false_calls
