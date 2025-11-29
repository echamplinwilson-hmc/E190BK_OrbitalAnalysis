import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta

# -------------------- Load SPICE kernels --------------------
spice.furnsh(r'C:\Users\elean\OneDrive - Harvey Mudd College\Documents\Spacecraft\data\bepiMeta.tm')
ck_file = r'C:\Users\elean\OneDrive - Harvey Mudd College\Documents\Spacecraft\naif_spice\bepi_kernels\ck\bc_mpo_sc_fsp_00208_20181020_20270328_f20181127_v10.bc'
sclk_file = r'C:\Users\elean\OneDrive - Harvey Mudd College\Documents\Spacecraft\naif_spice\bepi_kernels\sclk\bc_mpo_step_20181020.tsc'
spice.furnsh(ck_file)
spice.furnsh(sclk_file)

# -------------------- Parameters --------------------
MOI_DATE = "2026-11-26"
et_moi = spice.str2et(MOI_DATE)

# -------------------- Pre-MOI --------------------
et_start_pre = spice.str2et("2026-10-01")
et_end_pre   = spice.str2et("2026-11-25")
STEP_PRE = 3600  # 1 hour

times_pre = np.arange(et_start_pre, et_end_pre, STEP_PRE)
pos_pre_km, _ = spice.spkpos("-121", times_pre, "J2000", "NONE", "199")
pos_pre = np.array(pos_pre_km, dtype=float)
r_pre = np.linalg.norm(pos_pre, axis=1)
r_pre = np.maximum(r_pre, 1e-3)
dates_pre = [datetime(2000,1,1) + timedelta(seconds=(t-spice.str2et("2000-01-01T00:00:00"))) for t in times_pre]

# -------------------- Post-MOI --------------------
STEP_POST = 600  # 10 min
POST_MOI_DURATION_DAYS = 30
POST_MOI_DURATION = POST_MOI_DURATION_DAYS * 24 * 3600
et_start_post = et_moi
et_end_post = et_moi + POST_MOI_DURATION

times_post = np.arange(et_start_post, et_end_post, STEP_POST)
pos_post_km, _ = spice.spkpos("-121", times_post, "J2000", "NONE", "199")
pos_post = np.array(pos_post_km, dtype=float)
r_post = np.linalg.norm(pos_post, axis=1)
r_post = np.maximum(r_post, 1e-3)
dates_post = [datetime(2000,1,1) + timedelta(seconds=(t-spice.str2et("2000-01-01T00:00:00"))) for t in times_post]

peri_idx = np.argmin(r_post)
apo_idx = np.argmax(r_post)

# -------------------- Figure --------------------
fig = plt.figure(figsize=(12,10))

# Top panel: 2D distance with log scale
ax1 = fig.add_subplot(211)
ax1.plot(dates_pre, r_pre, color='blue', label='Distance Pre-MOI (Approach)')
ax1.plot(dates_post, r_post, color='green', label='Distance Post-MOI (Orbit)')
ax1.axvline(datetime(2026,11,26), color='red', linestyle='--', label='MOI')
ax1.scatter(dates_post[peri_idx], r_post[peri_idx], color='red', s=60, label='Periapsis')
ax1.scatter(dates_post[apo_idx], r_post[apo_idx], color='orange', s=60, label='Apoapsis')
ax1.set_xlabel('Date')
ax1.set_ylabel('Distance from Mercury [km]')
ax1.set_yscale('log')
ax1.set_title('MPO Distance from Mercury: Approach and Orbit (Log Scale)')
ax1.grid(True, which="both", ls="--")
ax1.legend()

# Bottom panel: 3D orbit
from mpl_toolkits.mplot3d import Axes3D
ax2 = fig.add_subplot(212, projection='3d')
line, = ax2.plot([], [], [], color='blue', label='MPO orbit')
point, = ax2.plot([], [], [], 'ro', label='MPO Current Position')

# Periapsis / Apoapsis
ax2.scatter(pos_post[peri_idx,0], pos_post[peri_idx,1], pos_post[peri_idx,2],
            color='red', s=80, label='Periapsis')
ax2.scatter(pos_post[apo_idx,0], pos_post[apo_idx,1], pos_post[apo_idx,2],
            color='orange', s=80, label='Apoapsis')

# Mercury
ax2.scatter(0,0,0,color='orange', s=200,label='Mercury')

ax2.set_xlabel('X [km]')
ax2.set_ylabel('Y [km]')
ax2.set_zlabel('Z [km]')
ax2.set_title('MPO Mercury Orbit Post-MOI (Animated)')
ax2.legend()

margin = 2000
ax2.set_xlim(np.min(pos_post[:,0])-margin, np.max(pos_post[:,0])+margin)
ax2.set_ylim(np.min(pos_post[:,1])-margin, np.max(pos_post[:,1])+margin)
ax2.set_zlim(np.min(pos_post[:,2])-margin, np.max(pos_post[:,2])+margin)

def update(num):
    line.set_data(pos_post[:num,0], pos_post[:num,1])
    line.set_3d_properties(pos_post[:num,2])
    point.set_data([pos_post[num,0]], [pos_post[num,1]])
    point.set_3d_properties([pos_post[num,2]])
    return line, point

ani = FuncAnimation(fig, update, frames=len(times_post), interval=50, blit=True)

plt.tight_layout()
plt.show()

# -------------------- Clear SPICE --------------------
spice.kclear()
