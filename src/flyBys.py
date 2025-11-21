import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt

# load kernels
spice.furnsh('C:\\Users\\elean\\OneDrive - Harvey Mudd College\\Documents\\Spacecraft\\data\\bepiMeta.tm')
print("Loaded kernels:", spice.ktotal('ALL'))

# times / dates
utc_start = "Oct 22, 2018"
utc_end   = "Mar 27, 2025"
et_start = spice.str2et(utc_start)
et_end   = spice.str2et(utc_end)

times = np.arange(et_start, et_end, 2e5)   # coarse 200,000 s (~2.3 days)

# spacecraft and planet IDs
spacecraft_id = "-121"
planets = {"Mercury": "199", "Venus": "299", "Earth": "399"}

# sun positions
pos_sc_sun, _ = spice.spkpos(spacecraft_id, times, 'J2000', 'NONE', '10')
pos_sc_sun = np.array(pos_sc_sun)

planet_positions = {}
for planet, pid in planets.items():
    pos, _ = spice.spkpos(pid, times, 'J2000', 'NONE', '10')
    planet_positions[planet] = np.array(pos)

# distance to planets
distances = {}
for planet, pos in planet_positions.items():
    distances[planet] = np.linalg.norm(pos_sc_sun - pos, axis=1)

# find flybys
howClose = 500000   # km threshold
flyby_intervals = {planet: [] for planet in planets}

for planet, dist_array in distances.items():
    in_flyby = False
    start_idx = None

    for i in range(len(dist_array)):
        if not in_flyby and dist_array[i] < howClose:
            in_flyby = True
            start_idx = i
        
        if in_flyby and dist_array[i] >= howClose:
            in_flyby = False
            end_idx = i
            flyby_intervals[planet].append((start_idx, end_idx))

    if in_flyby:
        flyby_intervals[planet].append((start_idx, len(dist_array)-1))

# find index for flybys
flyby_CA = {planet: [] for planet in planets}

for planet, intervals in flyby_intervals.items():
    for (i0, i1) in intervals:
        segment = distances[planet][i0:i1]
        min_rel_idx = np.argmin(segment)
        ca_idx = i0 + min_rel_idx
        flyby_CA[planet].append(ca_idx)

# plot flybys
for planet, intervals in flyby_intervals.items():
    print(f"\n=== {planet} Flybys ===")

    pid = planets[planet]

    for (i0, i1), ca_idx in zip(intervals, flyby_CA[planet]):

        et_ca = times[ca_idx]

        # better sampling for up close 
        window = 5 * 86400      # 5 days
        dt = 1000               # 17 mins

        t_fine = np.arange(et_ca - window, et_ca + window, dt)

        # High-res planet-centered distance
        sc_pos, _ = spice.spkpos(spacecraft_id, t_fine, 'J2000', 'NONE', pid)
        pl_pos, _ = spice.spkpos(pid, t_fine, 'J2000', 'NONE', '10')

        sc_pos = np.array(sc_pos)
        pl_pos = np.array(pl_pos)

        d_fine = np.linalg.norm(sc_pos - pl_pos, axis=1)

        # Relative time (days)
        t_rel_days = (t_fine - et_ca) / 86400.0

        # Print flyby info
        ca_dist = np.min(d_fine)
        print(f"Flyby at {spice.et2utc(et_ca, 'C', 3)}  |  {ca_dist:.1f} km")

        # Plot 
        plt.figure(figsize=(8,4))
        plt.plot(t_rel_days, d_fine, label="Distance (km)")
        plt.axvline(0, color='red', linestyle='--', label="Closest Approach")

        plt.title(f"{planet} Flyby — Distance vs Time\nFlyby: {spice.et2utc(et_ca, 'C', 3)}")
        plt.xlabel("Time relative to Flyby (days)")
        plt.ylabel("Distance (km)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

spice.kclear()
