import os

path = r"C:\Users\elean\OneDrive - Harvey Mudd College\Documents\Spacecraft\naif_spice\bepi_kernels\ck\bc_mpo_sc_fcp_00217_20181020_20260111_f20181127_v02.bc"
print(os.path.exists(path))

'''window = 5 * 24 * 3600

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
        plt.show()'''