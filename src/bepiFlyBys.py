import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt

# Load kernels
spice.furnsh(r"C:\Users\elean\OneDrive - Harvey Mudd College\Documents\Spacecraft\data\bepiMeta.tm")
print("Loaded kernels:", spice.ktotal('ALL'))

# Times
utc_start = "Oct 22 2018"
utc_end   = "Mar 27 2025"

t0 = spice.str2et(utc_start)
t1 = spice.str2et(utc_end)
times = np.arange(t0, t1, 3600)     # 1 hour steps

# IDs
planets = {
    "Mercury": "199",
    "Venus":   "299",
    "Earth":   "399"
}
sc_id = "-121"

# Get positions to Sun
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

# Flyby detection
threshold = 50e3 # 50 km

min_flyby_separation = 5 * 24 * 3600    # still fine to keep

flybys = {}

# Find flybys
for planet in planets:
    d = distances[planet]

    mask = d < threshold
    events = []
    in_event = False
    start = 0

    for i, flag in enumerate(mask):
        if flag and not in_event:
            in_event = True
            start = i

        if in_event and (not flag or i == len(mask) - 1):
            end = i

            # we removed the duration check here
            idx = start + np.argmin(d[start:end])
            events.append(idx)

            in_event = False

    flybys[planet] = events

# results
print("\n============================")
print("Detected Flybys")
print("============================")

for planet, idxs in flybys.items():
    print(f"\n{planet} = {len(idxs)} flybys")
    for idx in idxs:
        print(f"   {utc[idx]}   {distances[planet][idx]/1000:,.0f} km")

# plots
window = 5 * 24 * 3600

for planet, idxs in flybys.items():
    d = distances[planet]
    for j, ci in enumerate(idxs):
        tmin = times[ci] - window
        tmax = times[ci] + window

        sel = (times >= tmin) & (times <= tmax)

        plt.figure(figsize=(9,4))
        plt.plot(utc[sel], d[sel]/1e3)   # km is easier scale
        plt.scatter([utc[ci]], [d[ci]/1e3], c='red', s=60)

        plt.yscale("log")   # <-- ✨ log y-axis

        # limit how many x labels show
        plt.xticks(plt.xticks()[0][::12], rotation=25)

        plt.title(f"{planet} Flyby #{j+1}\nClosest = {utc[ci]}")
        plt.ylabel("Distance (km, log scale)")
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


spice.kclear()
