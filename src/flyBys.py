import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt

# Load kernels
spice.furnsh('C:\\Users\\elean\\OneDrive - Harvey Mudd College\\Documents\\Spacecraft\\data\\bepiMeta.tm')
print("Loaded kernels:", spice.ktotal('ALL'))

# Time range
utc_start = "Oct 22, 2018"
utc_end   = "Mar 27, 2025"
et_start = spice.str2et(utc_start)
et_end   = spice.str2et(utc_end)
times = np.arange(et_start, et_end, 2e5)

# all da IDs
spacecraft_id = '-121'
planets = {'Mercury': '199', 'Venus': '299', 'Earth': '399'}
colors = {'Mercury':'orange', 'Venus':'green', 'Earth':'blue'}


# all da positions relative to Sun
pos_mpo_sun, _ = spice.spkpos(spacecraft_id, times, 'J2000', 'NONE', '10')
pos_mpo_sun = np.array(pos_mpo_sun)

planet_positions = {}
for planet, pid in planets.items():
    pos, _ = spice.spkpos(pid, times, 'J2000', 'NONE', '10')
    planet_positions[planet] = np.array(pos)

# displacements
distances = {}
for planet, pos in planet_positions.items():
    distances[planet] = np.linalg.norm(pos_mpo_sun - pos, axis=1)
 
# get all the flybys    
flybys = {}
for planet, dist_array in distances.items():
    flybys[planet] = dist_array[dist_array < 5000000]

# UTC strings 
utc_dates = [spice.et2utc(et, 'C', 0) for et in times]

# Plot displacements vs time 
fig2d, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

for i, planet in enumerate(planets):
    axs[i].plot(utc_dates, flybys[planet]/1e6, color=colors[planet], lw=1.5) 
    axs[i].set_ylabel(f'{planet} Distance (million km)')
    axs[i].grid(True)
    axs[i].set_title(f'BepiColombo Distance to {planet}')

axs[2].set_xlabel('Date')
plt.tight_layout()
spice.kclear()
plt.show()
