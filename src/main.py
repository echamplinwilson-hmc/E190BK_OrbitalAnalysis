import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt

spice.furnsh('C:\\Users\\elean\\OneDrive - Harvey Mudd College\\Documents\\Spacecraft\\data\\cassMeta.tm')
print(spice.ktotal('ALL'))


utc_start = "Jun 20, 2004"
utc_end = "Dec 1, 2005"
et_start = spice.str2et(utc_start)
et_end = spice.str2et(utc_end)

times = list(range(int(et_start), int(et_end), 4000)) 
positions, lightTimes = spice.spkpos('CASSINI', times, 'J2000', 'NONE', 'SATURN BARYCENTER')

spice.kclear()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:,0], positions[:,1], positions[:,2])
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Cassini Trajectory from June 2004 to December 2005')
plt.show()