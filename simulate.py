"""
simulate.py - Create a realistic "unrolled" wall-thickness map of a well casing.

An ultrasonic imaging tool scans the pipe all the way around as it moves down
the well. If you cut the pipe open along one side and lay it flat, you get a
2D map:
    rows    = depth along the well (metres)
    columns = angle around the pipe (degrees, 0-359)
    value   = remaining wall thickness (mm)

This file builds that map and adds known defects, so we can later check
whether our detection code finds them (a quality check on our own method).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Pipe and scan settings (a common 7-inch production casing)
# ---------------------------------------------------------------------------
OUTER_DIAMETER_MM = 177.8      # 7 inches x 25.4
NOMINAL_WALL_MM = 9.19         # original (design) wall thickness
TOP_DEPTH_M = 1200.0           # start of the scanned interval
INTERVAL_M = 60.0              # length of pipe scanned
DEPTH_STEP_M = 0.01            # one reading every 1 cm along the pipe (high resolution)
ANGLE_STEP_DEG = 1.0           # one reading every 1 degree around the pipe
JOINT_LENGTH_M = 12.0          # a connection (collar) every ~12 m
COLLAR_LENGTH_M = 0.25         # collars are short, thicker sections

N_DEPTH = int(round(INTERVAL_M / DEPTH_STEP_M))  # 6000 rows
N_ANGLE = int(360 / ANGLE_STEP_DEG)             # 360 columns


def depth_to_row(depth_m):
    """Convert a well depth (m) to a row index in the map."""
    return int(round((depth_m - TOP_DEPTH_M) / DEPTH_STEP_M))


def add_pit_or_patch(thk, depth_m, angle_deg, length_m, width_deg, loss_frac):
    """
    Remove metal in a smooth, bowl-shaped area (deepest in the middle).
    loss_frac = fraction of the nominal wall lost at the deepest point.
    A 2D Gaussian shape gives realistic, rounded corrosion.
    """
    rows = np.arange(N_DEPTH)[:, None]
    cols = np.arange(N_ANGLE)[None, :]
    r0, c0 = depth_to_row(depth_m), angle_deg / ANGLE_STEP_DEG
    sr = (length_m / DEPTH_STEP_M) / 4.0          # spread along depth
    sc = (width_deg / ANGLE_STEP_DEG) / 4.0       # spread around pipe
    shape = np.exp(-(((rows - r0) / sr) ** 2 + ((cols - c0) / sc) ** 2) / 2)
    thk -= shape * loss_frac * NOMINAL_WALL_MM


def add_groove(thk, top_m, bottom_m, angle_deg, width_deg, loss_frac):
    """Long, narrow wear track along one side of the pipe (e.g. rod wear)."""
    r1, r2 = depth_to_row(top_m), depth_to_row(bottom_m)
    cols = np.arange(N_ANGLE)
    sc = (width_deg / ANGLE_STEP_DEG) / 4.0
    profile = np.exp(-(((cols - angle_deg / ANGLE_STEP_DEG) / sc) ** 2) / 2)
    thk[r1:r2, :] -= profile[None, :] * loss_frac * NOMINAL_WALL_MM


def simulate(seed=42):
    """
    Build the thickness map.
    Returns:
        thickness  - 2D array (depth x angle) in mm, with NaN where data is missing
        truth      - list of the defects we planted (the "answer key")
        depths_m   - depth of each row
    """
    rng = np.random.default_rng(seed)
    depths_m = TOP_DEPTH_M + np.arange(N_DEPTH) * DEPTH_STEP_M

    # 1) Start with a perfect pipe at nominal thickness, plus small real-world
    #    manufacturing variation (pipe wall is never perfectly even).
    thk = np.full((N_DEPTH, N_ANGLE), NOMINAL_WALL_MM)
    thk += rng.normal(0, 0.08, size=thk.shape)

    # 2) Collars: every joint has a short, thicker connection. These are NORMAL
    #    geometry - a good analyst must not report them as defects.
    collar_depths = np.arange(TOP_DEPTH_M + 6.0, TOP_DEPTH_M + INTERVAL_M, JOINT_LENGTH_M)
    for d in collar_depths:
        r = depth_to_row(d)
        thk[r:r + int(COLLAR_LENGTH_M / DEPTH_STEP_M), :] += 3.0

    # 3) Planted defects - the answer key we will check our detection against.
    truth = [
        dict(type="Pitting",            depth_m=1209.4, angle_deg=95,  loss_pct=45, length_m=0.20),
        dict(type="Pitting",            depth_m=1209.9, angle_deg=110, loss_pct=30, length_m=0.15),
        dict(type="General corrosion",  depth_m=1224.0, angle_deg=200, loss_pct=25, length_m=2.50),
        dict(type="Wear groove",        depth_m=1240.0, angle_deg=300, loss_pct=20, length_m=8.00),
        dict(type="Severe pit",         depth_m=1251.2, angle_deg=45,  loss_pct=65, length_m=0.25),
    ]
    add_pit_or_patch(thk, 1209.4, 95, 0.20, 12, 0.45)   # pit
    add_pit_or_patch(thk, 1209.9, 110, 0.15, 10, 0.30)  # nearby pit (a cluster)
    add_pit_or_patch(thk, 1224.0, 200, 2.50, 70, 0.25)  # broad general corrosion
    add_groove(thk, 1236.0, 1244.0, 300, 8, 0.20)        # 8 m long wear track
    add_pit_or_patch(thk, 1251.2, 45, 0.25, 14, 0.65)   # deep, severe pit

    # 4) Measurement noise: random speckle, like a real ultrasound reading.
    thk += rng.normal(0, 0.25, size=thk.shape)
    speckle = rng.random(thk.shape) < 0.002              # 0.2% "spike" readings
    thk[speckle] -= rng.uniform(2, 4, size=speckle.sum())  # fake thin spots!

    # 5) A data-quality problem: a short interval with no echo (e.g. gas in
    #    the fluid). Missing readings are stored as NaN.
    r1, r2 = depth_to_row(1232.0), depth_to_row(1232.8)
    thk[r1:r2, 150:250] = np.nan

    return thk, truth, depths_m, collar_depths
