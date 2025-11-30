import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# kernels
spice.furnsh(r'C:\Users\elean\OneDrive - Harvey Mudd College\Documents\Spacecraft\data\bepiMeta.tm')
ck_file = r'C:\Users\elean\OneDrive - Harvey Mudd College\Documents\Spacecraft\naif_spice\bepi_kernels\ck\bc_mpo_sc_fsp_00208_20181020_20270328_f20181127_v10.bc'
sclk_file = r'C:\Users\elean\OneDrive - Harvey Mudd College\Documents\Spacecraft\naif_spice\bepi_kernels\sclk\bc_mpo_step_20181020.tsc'
spice.furnsh(ck_file)
spice.furnsh(sclk_file)

# constants and times
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
ET_SEP = spice.str2et("2026-10-01")        # MPO/MTM separation (approx)
ET_MOI = spice.str2et("2026-11-30")        # Mercury Orbit Insertion event (approx)

# spacecraft properties (ish)
mass_MCS = 4100.0
mass_MPO = 1150.0
dims_MCS = [3.9,3.6,6.3]  # approximate overall dimensions [m]
dims_MPO = [2.4,2.2,1.7]  # approximate overall dimensions [m]

# thrusters 
# Ion thrusters
ION_COUNT = 4
ION_THRUST = 0.145       # N per thruster
ION_ISP = 4300
ION_IMPULSE_PER_FIRING = 23.7e-3  # N*s per thruster per desat
LEVER_ION = 1.0          # m

# Chemical thruster(s) (used for MOI event ONLYYY)
CHEM_LARGE_THRUST = 22.0
CHEM_LARGE_COUNT = 4
CHEM_ACS_THRUST = 10.0
CHEM_ACS_COUNT = 4
CHEM_ISP = 200
CHEM_BOOSTER_PROP_USED = 300.0  # kg used for MOI
LEVER_CHEM_LARGE = 1.0
LEVER_CHEM_ACS = 0.7

# desats
DT_CRUISE = 12*3600  # seconds between desats (12h)

# utilities 
# based on box inertia calculations with approximate dimensions from esa
# mostly because its not avalilable to public yet
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
    lever=np.array([0,3.75,3.15]) if et<ET_SEP else np.array([0,15,6]) # accoutns for "lever arm" sim. H shape of sat
    F_vec = -F_mag*r_sun_vec/r_sun
    return np.cross(lever,F_vec)

# SPK positions
pos_sc_mer_km,_ = spice.spkpos(SC_ID, times,"J2000","NONE",MERCURY)
pos_mer_sun_km,_ = spice.spkpos(MERCURY, times,"J2000","NONE",SUN)
pos_sc_mer = np.array(pos_sc_mer_km)*1000
pos_mer_sun = np.array(pos_mer_sun_km)*1000
pos_sc_sun = pos_sc_mer + pos_mer_sun

# thruster dataclass w/ attributes
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
        # F_used is thrust required (N) *for the entire group* (we will pass group-level F if needed)
        if F_used<=0: return 0.0
        frac=min(1.0,F_used/self.total_thrust)
        mdot=(self.total_thrust*frac)/(self.isp_s*g0)
        return mdot*duration

ION_GROUP = ThrusterGroup("ion",ION_THRUST,ION_ISP,ION_COUNT,LEVER_ION,ION_IMPULSE_PER_FIRING)
CHEM_LARGE_GROUP = ThrusterGroup("chem_large",CHEM_LARGE_THRUST,CHEM_ISP,CHEM_LARGE_COUNT,LEVER_CHEM_LARGE)
CHEM_ACS_GROUP = ThrusterGroup("chem_acs",CHEM_ACS_THRUST,CHEM_ISP,CHEM_ACS_COUNT,LEVER_CHEM_ACS)

# dynamics initial conditions
mass = mass_MCS
I_current = box_I(mass_MCS,dims_MCS)
sep_done = False
mois_done = False

H_rw = np.zeros((len(times),3))
torq = np.zeros((len(times),3))
mass_over_time = np.zeros(len(times))
H=np.zeros(3)
total_ion_used=0.0

# arrays for thrust, duty cycle, and delta-V
ion_req = np.zeros(len(times))     # required per-ion-thruster thrust [N]
chem_req = np.zeros(len(times))    # required per-chem-thruster thrust [N]
ion_duty = np.zeros(len(times))    # duty fraction (0..1) for ion thruster
chem_duty = np.zeros(len(times))   # duty fraction for chemical thruster
ion_deltaV_events = []             # store delta-V values (m/s) per ion desat event
chem_deltaV_events = []            # store delta-V values (m/s) for MOI (may be single value)
ion_mass_used_events = []          # mass used per ion desat
chem_mass_used = 0.0

last_desat_et = times[0]

for i,t in enumerate(times):
    # update mass and inertia at separation
    if (not sep_done) and t>=ET_SEP:
        mass=mass_MPO
        I_current=box_I(mass,dims_MPO)
        sep_done=True

    # torques (gravity gradient + SRP)
    r_body=to_body_frame(pos_sc_mer[i],t)
    s_body=to_body_frame(pos_sc_sun[i],t)
    T=gg_torque(r_body,I_current)+srp_torque(s_body,t)
    torq[i]=T

    # integrate reaction wheel (simple integrator)
    H += T*STEP
    H_rw[i]=H

    # apply MOI chemical burn once (we will compute burn duration and spread chem_req over indices)
    if (not mois_done) and t>=ET_MOI:
        # record mass before MOI
        m_before = mass
        # subtract chemical fuel used
        mass = max(0.0, mass - CHEM_BOOSTER_PROP_USED)
        chem_mass_used = min(CHEM_BOOSTER_PROP_USED, m_before)  # actual used (if mass limited)
        # update inertia after burn
        I_current = box_I(mass, dims_MPO if sep_done else dims_MCS)
        mois_done=True

        # compute delta-V from chemical booster (with da rocket equation!!!)
        if m_before > mass and mass > 0:
            dv_chem = CHEM_LARGE_GROUP.isp_s * g0 * np.log(m_before / mass)
        else:
            dv_chem = 0.0
        chem_deltaV_events.append(dv_chem)

        # compute burn duration assuming CHEM_LARGE_GROUP total thrust (large thrusters)
        total_chem_thrust = CHEM_LARGE_THRUST * CHEM_LARGE_COUNT
        mdot_total = total_chem_thrust / (CHEM_ISP * g0) if (CHEM_ISP*g0) != 0 else np.inf
        if mdot_total > 0:
            chem_burn_duration = chem_mass_used / mdot_total
        else:
            chem_burn_duration = 0.0

        # per-thruster thrust (we assume large main chemical thrusters used alone for entry)
        per_chem_thr = CHEM_LARGE_THRUST

        # fill chem_req and chem_duty for indices during the burn window
        end_et = t + chem_burn_duration
        # find indices j where times[j] >= t and times[j] < end_et
        j_start = i
        # ensure j_end is at least j_start
        j_end = np.searchsorted(times, end_et, side='right')
        # set per-thruster thrust and duty (duty = fraction of max thrust used, here assumed 1)
        for j in range(j_start, min(j_end, len(times))):
            chem_req[j] = per_chem_thr
            chem_duty[j] = min(1.0, per_chem_thr / CHEM_LARGE_THRUST)  # assumed will be 1.0

    # Gradual ion desaturation every DT_CRUISE before MOI
    if t < ET_MOI and t - last_desat_et >= DT_CRUISE:
        # compute required torque average and convert to thrust
        H_norm = np.linalg.norm(H_rw[i])
        T_avg  = H_norm / DT_CRUISE

        # group-level force required to generate torque about lever with all ion thrusters acting
        # torque = lever_total_effective * F_group. For simple model: F_group = T_avg / lever
        # per-thruster required = F_group / count
        F_group = T_avg / ION_GROUP.lever_m
        per_thr_required = F_group / ION_GROUP.count

        # record per-thruster required thrust at this time step
        ion_req[i] = per_thr_required

        # duty cycle as fraction of max thruster thrust
        ion_duty[i] = min(1.0, per_thr_required / ION_GROUP.thrust_N)

        # compute fuel used for the desat event using ThrusterGroup.fuel_for_duration,
        # but the method expects F_used to be the group-level applied thrust; pass F_group.
        fuel = ION_GROUP.fuel_for_duration(F_group, DT_CRUISE)
        total_ion_used += fuel

        # delta-V from the ion fuel used (rocket equation!!!!!!!): compute mass before/after
        m_before = mass
        mass = max(0.0, mass - fuel)
        ion_mass_used_events.append(fuel)
        if m_before > mass and mass > 0:
            dv_ion = ION_GROUP.isp_s * g0 * np.log(m_before / mass)
        else:
            dv_ion = 0.0
        ion_deltaV_events.append(dv_ion)

        # reset wheels after desat
        H_rw[i] = np.zeros(3)
        last_desat_et = t
        # update inertia after mass change
        I_current = box_I(mass, dims_MPO if sep_done else dims_MCS)

    mass_over_time[i] = mass

# Final summary values
final_mass = mass_over_time[-1]
N_desats = int((ET_MOI - times[0]) / DT_CRUISE)

total_ion_deltaV = sum(ion_deltaV_events)
total_chem_deltaV = sum(chem_deltaV_events)
total_deltaV = total_ion_deltaV + total_chem_deltaV

print("------------- MISSION SUMMARY -------------")
print(f"Total ion propellant used [kg]: {total_ion_used:.6f}")
print(f"Chemical booster prop used for MOI [kg]: {chem_mass_used:.3f}")
print(f"Number of desaturation events (expected): {N_desats}")
print(f"Final spacecraft mass [kg]: {final_mass:.3f}")
print(f"Total delta-V from ion desats [m/s]: {total_ion_deltaV:.6f}")
print(f"Total delta-V from chemical MOI [m/s]: {total_chem_deltaV:.6f}")
print(f"Total mission delta-V [m/s]: {total_deltaV:.6f}")
print("------------------------------------------")

# plots
years = (times - times[0])/(86400*365)

# Torque components
plt.figure(figsize=(11,4))
plt.plot(years,torq[:,0],label='Tx')
plt.plot(years,torq[:,1],label='Ty')
plt.plot(years,torq[:,2],label='Tz')
plt.title("Disturbance Torque Components")
plt.xlabel("Years since launch"); plt.ylabel("Torque [N m]"); plt.legend(); plt.grid(True)

# Accumulated reaction wheel H
plt.figure(figsize=(11,4))
plt.plot(years,H_rw[:,0],label='Hx')
plt.plot(years,H_rw[:,1],label='Hy')
plt.plot(years,H_rw[:,2],label='Hz')
plt.title("Accumulated Reaction Wheel Angular Momentum")
plt.xlabel("Years since launch"); plt.ylabel("H [N m s]"); plt.legend(); plt.grid(True)

# Mass plot
plt.figure(figsize=(11,4))
plt.plot(years, mass_over_time, color='purple')
plt.title("Spacecraft Mass Over Mission")
plt.xlabel("Years since launch")
plt.ylabel("Mass [kg]")
plt.grid(True)
plt.axvline((ET_SEP - times[0])/(86400*365), color='orange', linestyle='--', label='MPO/MTM Separation')
plt.axvline((ET_MOI - times[0])/(86400*365), color='red', linestyle='--', label='Mercury Orbit Insertion (MOI)')
plt.legend()

# Duty cycle plot
plt.figure(figsize=(11,4))
plt.plot(years, ion_duty, label='Ion thruster duty')
plt.plot(years, chem_duty, label='Chemical thruster duty')
plt.title("Thruster Duty Cycle Over Time")
plt.xlabel("Years since launch")
plt.ylabel("Duty cycle (fraction of max thrust)")
plt.grid(True)
plt.legend()

plt.show()

spice.kclear()