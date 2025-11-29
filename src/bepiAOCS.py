import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# -------------------- SPICE kernels --------------------
spice.furnsh(r'C:\Users\elean\OneDrive - Harvey Mudd College\Documents\Spacecraft\data\bepiMeta.tm')
ck_file = r'C:\Users\elean\OneDrive - Harvey Mudd College\Documents\Spacecraft\naif_spice\bepi_kernels\ck\bc_mpo_sc_fsp_00208_20181020_20270328_f20181127_v10.bc'
sclk_file = r'C:\Users\elean\OneDrive - Harvey Mudd College\Documents\Spacecraft\naif_spice\bepi_kernels\sclk\bc_mpo_step_20181020.tsc'
spice.furnsh(ck_file)
spice.furnsh(sclk_file)

# -------------------- Constants & times --------------------
SC_ID = "-121"
MERCURY = "199"
SUN = "10"

UTC_START = "2018-10-22"
UTC_END   = "2026-12-22"
STEP = 20000  # seconds

GM = 2.2032e13  # Mercury mu [m^3/s^2]
g0 = 9.80665

et_start = spice.str2et(UTC_START)
et_end   = spice.str2et(UTC_END)
times = np.arange(et_start, et_end, STEP)

# Mission epochs
ET_SEP = spice.str2et("2026-10-01")        # MPO/MTM separation
ET_MOI = spice.str2et("2026-11-30")        # Mercury Orbit Insertion event

# -------------------- Spacecraft config --------------------
mass_MCS = 4100.0
mass_MPO = 1150.0
dims_MCS = [3.5,2.5,3.0]
dims_MPO = [2.4,2.2,1.7]

# -------------------- Thruster config --------------------
# Ion thrusters
ION_COUNT = 4
ION_THRUST = 0.145       # N per thruster
ION_ISP = 4300
ION_IMPULSE_PER_FIRING = 23.7e-3  # N*s per thruster per desat
LEVER_ION = 1.0          # m

# Chemical thrusters (used for MOI event only)
CHEM_LARGE_THRUST = 22.0
CHEM_LARGE_COUNT = 4
CHEM_ACS_THRUST = 10.0
CHEM_ACS_COUNT = 4
CHEM_ISP = 200
CHEM_BOOSTER_PROP_USED = 300.0
LEVER_CHEM_LARGE = 1.0
LEVER_CHEM_ACS = 0.7

# -------------------- Desat schedule --------------------
DT_CRUISE = 12*3600  # 12 h between ion desats

# -------------------- Utilities --------------------
def box_I(m,d):
    x,y,z=d
    return np.diag([(1/12)*m*(y*y+z*z),
                    (1/12)*m*(x*x+z*z),
                    (1/12)*m*(x*x+y*y)])

def to_body_frame(vec, et):
    try:
        R = np.array(spice.pxform("J2000","MPO_SPACECRAFT",et))
    except:
        R = np.eye(3)
    return R @ vec

def gg_torque(r_body,I):
    r=np.linalg.norm(r_body)
    if r==0: return np.zeros(3)
    u=r_body/r
    return 3*GM/r**3 * np.cross(I@u,u)

def srp_torque(r_sun_vec,et):
    P0=4.56e-6
    r_sun=np.linalg.norm(r_sun_vec)
    if r_sun==0: return np.zeros(3)
    F_mag = P0*(1/(r_sun/1.496e11)**2)*12*1.2
    lever=np.array([0,3.75,3.15]) if et<ET_SEP else np.array([0,15,6])
    F_vec = -F_mag*r_sun_vec/r_sun
    return np.cross(lever,F_vec)

# -------------------- SPK positions --------------------
pos_sc_mer_km,_ = spice.spkpos(SC_ID, times,"J2000","NONE",MERCURY)
pos_mer_sun_km,_ = spice.spkpos(MERCURY, times,"J2000","NONE",SUN)
pos_sc_mer = np.array(pos_sc_mer_km)*1000
pos_mer_sun = np.array(pos_mer_sun_km)*1000
pos_sc_sun = pos_sc_mer + pos_mer_sun

# -------------------- Thruster group --------------------
@dataclass
class ThrusterGroup:
    name: str
    thrust_N: float
    isp_s: float
    count: int
    lever_m: float
    max_impulse_per_thr: float = None

    @property
    def total_thrust(self):
        return self.thrust_N*self.count

    def fuel_for_duration(self,F_used,duration):
        if F_used<=0: return 0.0
        frac=min(1.0,F_used/self.total_thrust)
        mdot=(self.total_thrust*frac)/(self.isp_s*g0)
        return mdot*duration

ION_GROUP = ThrusterGroup("ion",ION_THRUST,ION_ISP,ION_COUNT,LEVER_ION,ION_IMPULSE_PER_FIRING)
CHEM_LARGE_GROUP = ThrusterGroup("chem_large",CHEM_LARGE_THRUST,CHEM_ISP,CHEM_LARGE_COUNT,LEVER_CHEM_LARGE)
CHEM_ACS_GROUP = ThrusterGroup("chem_acs",CHEM_ACS_THRUST,CHEM_ISP,CHEM_ACS_COUNT,LEVER_CHEM_ACS)

# -------------------- Main dynamics --------------------
mass = mass_MCS
I_current = box_I(mass_MCS,dims_MCS)
sep_done = False
mois_done = False

H_rw = np.zeros((len(times),3))
torq = np.zeros((len(times),3))
mass_over_time = np.zeros(len(times))
H=np.zeros(3)
total_ion_used=0.0

# -------------------- Gradual ion desat every DT_CRUISE --------------------
last_desat_et = times[0]

for i,t in enumerate(times):
    # update mass and inertia at separation
    if (not sep_done) and t>=ET_SEP:
        mass=mass_MPO
        I_current=box_I(mass,dims_MPO)
        sep_done=True

    # apply MOI chemical burn once
    if (not mois_done) and t>=ET_MOI:
        mass=max(0.0,mass-CHEM_BOOSTER_PROP_USED)
        mois_done=True
        I_current=box_I(mass,dims_MPO)

    # torques
    r_body=to_body_frame(pos_sc_mer[i],t)
    s_body=to_body_frame(pos_sc_sun[i],t)
    T=gg_torque(r_body,I_current)+srp_torque(s_body,t)
    torq[i]=T

    # integrate reaction wheel
    H+=T*STEP
    H_rw[i]=H

    # Gradual ion desaturation every 12h before MOI
    if t<ET_MOI and t - last_desat_et >= DT_CRUISE:
        H_norm=np.linalg.norm(H_rw[i])
        T_avg = H_norm/DT_CRUISE
        F_req = T_avg/(2*ION_GROUP.lever_m)
        fuel = ION_GROUP.fuel_for_duration(F_req, DT_CRUISE)
        total_ion_used += fuel
        mass = max(0.0,mass-fuel)
        I_current=box_I(mass,dims_MPO if sep_done else dims_MCS)
        H_rw[i]=np.zeros(3)  # wheels reset after desat
        last_desat_et = t

    mass_over_time[i]=mass

# -------------------- Final summary --------------------
final_mass=mass_over_time[-1]
N_desats=int((ET_MOI - times[0])/DT_CRUISE)

print("========== MISSION SUMMARY ==========")
print("Total ion propellant used [kg]:", total_ion_used)
print("Chemical booster prop used for MOI [kg]:", CHEM_BOOSTER_PROP_USED)
print("Number of desaturation events:", N_desats)
print("Final spacecraft mass [kg]:", final_mass)
print("====================================")

# -------------------- Optional plots --------------------
plt.figure(figsize=(11,4))
plt.plot((times-times[0])/(86400*365),torq[:,0],label='Tx')
plt.plot((times-times[0])/(86400*365),torq[:,1],label='Ty')
plt.plot((times-times[0])/(86400*365),torq[:,2],label='Tz')
plt.title("Disturbance Torque Components")
plt.xlabel("Years since launch"); plt.ylabel("Torque [N m]"); plt.legend(); plt.grid(True)

plt.figure(figsize=(11,4))
plt.plot((times-times[0])/(86400*365),H_rw[:,0],label='Hx')
plt.plot((times-times[0])/(86400*365),H_rw[:,1],label='Hy')
plt.plot((times-times[0])/(86400*365),H_rw[:,2],label='Hz')
plt.title("Accumulated Reaction Wheel Angular Momentum")
plt.xlabel("Years since launch"); plt.ylabel("H [N m s]"); plt.legend(); plt.grid(True)
plt.show()

# -------------------- Plot spacecraft mass over mission --------------------
plt.figure(figsize=(11,4))
plt.plot((times - times[0])/(86400*365), mass_over_time, color='purple')
plt.title("Spacecraft Mass Over Mission")
plt.xlabel("Years since launch")
plt.ylabel("Mass [kg]")
plt.grid(True)

# Mark key events
plt.axvline((ET_SEP - times[0])/(86400*365), color='orange', linestyle='--', label='MPO/MTM Separation')
plt.axvline((ET_MOI - times[0])/(86400*365), color='red', linestyle='--', label='Mercury Orbit Insertion (MOI)')
plt.legend()
plt.show()

# -------------------- Clear SPICE --------------------
spice.kclear()
