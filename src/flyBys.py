import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Load kernels
# -------------------------------------------------------------
spice.furnsh(r"C:\Users\elean\OneDrive - Harvey Mudd College\Documents\Spacecraft\data\bepiMeta.tm")
print("Loaded kernels:", spice.ktotal('ALL'))

# -------------------------------------------------------------
# Time setup
# -------------------------------------------------------------
utc_start = "Oct 22 2018"
utc_end   = "Mar 27 2025"

t0 = spice.str2et(utc_start)
t1 = spice.str2et(utc_end)
times = np.arange(t0, t1, 2e5)     # ~2.3 days resolution

planets = {
    "Mercury": "199",
    "Venus":   "299",
    "Earth":   "399"
}
colors = {
    "Mercury": "orange",
    "Venus":   "green",
    "Earth":   "blue"
}
sc_id = "-121"

# -------------------------------------------------------------
# Get positions relative to Sun
# -------------------------------------------------------------
pos_sc, _ = spice.spkpos(sc_id, times, "J2000", "NONE", "10")
pos_sc = np.array(pos_sc)

positions = {}
for name, pid in planets.items():
    pos, _ = spice.spkpos(pid, times, "J2000", "NONE", "10")
    positions[name] = np.array(pos)

# compute distances
distances = {name: np.linalg.norm(pos_sc - positions[name], axis=1)
             for name in planets}

utc = np.array([spice.et2utc(t, 'C', 0) for t in times])

# -------------------------------------------------------------
# Flyby detection settings
# -------------------------------------------------------------
thresholds = {
    "Mercury": 2.0e7,
    "Venus":   1.0e7,
    "Earth":   1.0e6        
}

min_under_duration   = 3 * 24 * 3600     # must stay under threshold ≥ 3 days
min_flyby_separation = 30 * 24 * 3600    # flybys must be ≥ 30 days apart

flybys = {}

# -------------------------------------------------------------
# Find flyby candidates
# -------------------------------------------------------------
for planet in planets:
    d = distances[planet]
    th = thresholds[planet]

    mask = d < th
    events = []
    in_event = False
    start = 0

    for i, flag in enumerate(mask):
        if flag and not in_event:
            in_event = True
            start = i

        if in_event and (not flag or i == len(mask) - 1):
            end = i
            duration = times[end] - times[start]
            if duration > min_under_duration:
                idx = start + np.argmin(d[start:end])
                events.append(idx)
            in_event = False

    # merge events that are too close in time
    merged = []
    for idx in events:
        if not merged:
            merged.append(idx)
        elif times[idx] - times[merged[-1]] > min_flyby_separation:
            merged.append(idx)

    flybys[planet] = merged

# -------------------------------------------------------------
# Print results
# -------------------------------------------------------------
print("\n============================")
print("Detected Flybys")
print("============================")

for planet, idxs in flybys.items():
    print(f"\n{planet} = {len(idxs)} flybys")
    for idx in idxs:
        print(f"   {utc[idx]}   {distances[planet][idx]/1000:,.0f} km")

# -------------------------------------------------------------
# Plot zoomed windows for each flyby
# -------------------------------------------------------------
window = 5 * 24 * 3600

for planet, idxs in flybys.items():
    d = distances[planet]
    for j, ci in enumerate(idxs):
        tmin = times[ci] - window
        tmax = times[ci] + window

        sel = (times >= tmin) & (times <= tmax)

        plt.figure(figsize=(8,4))
        plt.plot(utc[sel], d[sel]/1e6, color=colors[planet])
        plt.scatter([utc[ci]], [d[ci]/1e6], c='red', s=60)
        plt.title(f"{planet} Flyby #{j+1}\nClosest = {utc[ci]}")
        plt.ylabel("Distance (million km)")
        plt.xticks(rotation=25)
        plt.grid()
        plt.tight_layout()
        plt.show()

spice.kclear()
