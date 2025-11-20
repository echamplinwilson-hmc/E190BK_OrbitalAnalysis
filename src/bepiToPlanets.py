import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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

# UTC strings 
utc_dates = [spice.et2utc(et, 'C', 0) for et in times]

# Plot displacements vs time 
fig2d, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

for i, planet in enumerate(planets):
    axs[i].plot(utc_dates, distances[planet]/1e6, color=colors[planet], lw=1.5)
    axs[i].set_ylabel(f'{planet} Distance (million km)')
    axs[i].grid(True)
    axs[i].set_title(f'BepiColombo Distance to {planet}')

axs[2].set_xlabel('Date')
plt.tight_layout()

# 3D Animation stuff --> https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
fig3d = plt.figure(figsize=(14,10))
ax = fig3d.add_subplot(111, projection='3d')

# Full trajectories
ax.plot(pos_mpo_sun[:,0], pos_mpo_sun[:,1], pos_mpo_sun[:,2], color='red', lw=1, label='BepiColombo')
for planet, pos in planet_positions.items():
    ax.plot(pos[:,0], pos[:,1], pos[:,2], color=colors[planet], lw=1, label=planet)

# Moving points
mpo_point, = ax.plot([], [], [], 'ro', markersize=6)
planet_points = {planet: ax.plot([], [], [], 'o', color=colors[planet], markersize=6)[0] for planet in planets}
lines = {planet: ax.plot([], [], [], color='gray', alpha=0.5)[0] for planet in planets}

# Dynamic labels and stuff --> https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
label_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12, verticalalignment='top')

ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('BepiColombo Trajectory Animation')
ax.legend()

# Axis limits
all_positions = np.vstack([pos_mpo_sun] + [p for p in planet_positions.values()])
ax.set_xlim(all_positions[:,0].min()*1.1, all_positions[:,0].max()*1.1)
ax.set_ylim(all_positions[:,1].min()*1.1, all_positions[:,1].max()*1.1)
ax.set_zlim(all_positions[:,2].min()*1.1, all_positions[:,2].max()*1.1)

# Animation update function --> https://www.youtube.com/watch?v=NhnowBBLtmo
def update(frame):
    # Spacecraft
    mpo_point.set_data([pos_mpo_sun[frame,0]], [pos_mpo_sun[frame,1]])
    mpo_point.set_3d_properties([pos_mpo_sun[frame,2]])

    # Planets and lines
    distances_frame = {}
    for planet, pos in planet_positions.items():
        planet_points[planet].set_data([pos[frame,0]], [pos[frame,1]])
        planet_points[planet].set_3d_properties([pos[frame,2]])

        # Line to planet thing
        lines[planet].set_data([pos_mpo_sun[frame,0], pos[frame,0]],
                               [pos_mpo_sun[frame,1], pos[frame,1]])
        lines[planet].set_3d_properties([pos_mpo_sun[frame,2], pos[frame,2]])

        # Distances
        distances_frame[planet] = np.linalg.norm(pos_mpo_sun[frame] - pos[frame])/1e6
        
    # Dynamic label with date + distances
    current_date = spice.et2utc(times[frame], 'C', 0)
    label_text.set_text(
        f"Date: {current_date}\n" +
        "\n".join([f"{planet}: {dist:.2f} million km" for planet, dist in distances_frame.items()])
    )

    return [mpo_point, label_text] + list(planet_points.values()) + list(lines.values())

# animation go
ani = FuncAnimation(fig3d, update, frames=len(times), interval=50, blit=True)

# Show / end
plt.show()
spice.kclear()
