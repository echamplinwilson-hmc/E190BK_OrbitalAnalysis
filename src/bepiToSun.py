import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt

# kernels
spice.furnsh('C:\\Users\\elean\\OneDrive - Harvey Mudd College\\Documents\\Spacecraft\\data\\bepiMeta.tm')
ids = spice.spkobj("naif_spice\\cassini_kernels\\spk\\bc_mpo_fcp_00206_20181020_20270407_v02.bsp")
print(ids)
print(list(ids))
print(spice.ktotal('ALL'))

# times 
utc_start = "Oct 22, 2018"
utc_end   = "Mar 27, 2025"
et_start = spice.str2et(utc_start)
et_end   = spice.str2et(utc_end)
times = np.arange(et_start, et_end, 20000)


# positions
# BepiColombo to Mercury
pos_mpo_mercury, _ = spice.spkpos('-121', times, 'J2000', 'NONE', '199')
pos_mpo_mercury = np.array(pos_mpo_mercury)

# Mercury to Sun
pos_mercury_sun, _ = spice.spkpos('199', times, 'J2000', 'NONE', '10')
pos_mercury_sun = np.array(pos_mercury_sun)

# BepiColombo to Sun
pos_mpo_sun = pos_mpo_mercury + pos_mercury_sun

# plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(pos_mpo_sun[:,0], pos_mpo_sun[:,1], pos_mpo_sun[:,2], color='blue', lw=1)
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('BepiColombo Trajectory Relative to the Sun')
ax.grid(True)

plt.show()
spice.kclear()