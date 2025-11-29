import spiceypy as spice
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime
import matplotlib.dates as mdates

# -------------------- SPICE kernels --------------------
spice.furnsh(r'C:\Users\elean\OneDrive - Harvey Mudd College\Documents\Spacecraft\data\bepiMeta.tm')
ck_file = r'C:\Users\elean\OneDrive - Harvey Mudd College\Documents\Spacecraft\naif_spice\bepi_kernels\ck\bc_mpo_sc_fsp_00208_20181020_20270328_f20181127_v10.bc'
sclk_file = r'C:\Users\elean\OneDrive - Harvey Mudd College\Documents\Spacecraft\naif_spice\bepi_kernels\sclk\bc_mpo_step_20181020.tsc'
spice.furnsh(ck_file)
spice.furnsh(sclk_file)

# -------------------- Constants --------------------
STEP = 86400  # 1 day integration
GM = 2.2032e13
g0 = 9.80665

mass_MCS = 4100.0
mass_MPO = 1150.0
dims_MCS = [3.5,2.5,3.0]
dims_MPO = [2.4,2.2,1.7]

ION_COUNT = 4
ION_THRUST = 0.145
ION_ISP = 4300
LEVER_ION = 1.0
DT_CRUISE = 24*3600  # 1 day desats

CHEM_BOOSTER_PROP_USED = 300.0  # kg

SC_ID = "-121"

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
    lever=np.array([0,15,6])
    F_vec = -F_mag*r_sun_vec/r_sun
    return np.cross(lever,F_vec)

@dataclass
class ThrusterGroup:
    thrust_N: float
    isp_s: float
    count: int
    lever_m: float

    @property
    def total_thrust(self):
        return self.thrust_N*self.count

    def fuel_for_duration(self,F_used,duration):
        if F_used<=0: return 0.0
        frac=min(1.0,F_used/self.total_thrust)
        mdot=(self.total_thrust*frac)/(self.isp_s*g0)
        return mdot*duration

ION_GROUP = ThrusterGroup(ION_THRUST,ION_ISP,ION_COUNT,LEVER_ION)

# -------------------- SPK positions --------------------
times_full = np.arange(spice.str2et("2026-09-15"), spice.str2et("2026-12-15")+STEP, STEP)
pos_sc_mer_km,_ = spice.spkpos(SC_ID, times_full,"J2000","NONE","199")
pos_mer_sun_km,_ = spice.spkpos("199", times_full,"J2000","NONE","10")
pos_sc_mer = np.array(pos_sc_mer_km)*1000
pos_mer_sun = np.array(pos_mer_sun_km)*1000
pos_sc_sun = pos_sc_mer + pos_mer_sun

# -------------------- Precompute torques for all times --------------------
torque_precompute_MCS = np.zeros((len(times_full),3))
for i,t in enumerate(times_full):
    r_body = to_body_frame(pos_sc_mer[i], t)
    s_body = to_body_frame(pos_sc_sun[i], t)
    I = box_I(mass_MCS,dims_MCS)
    torque_precompute_MCS[i] = gg_torque(r_body,I) + srp_torque(s_body,t)

torque_precompute_MPO = np.zeros((len(times_full),3))
for i,t in enumerate(times_full):
    r_body = to_body_frame(pos_sc_mer[i], t)
    s_body = to_body_frame(pos_sc_sun[i], t)
    I = box_I(mass_MPO,dims_MPO)
    torque_precompute_MPO[i] = gg_torque(r_body,I) + srp_torque(s_body,t)

# -------------------- Simulation --------------------
def simulate_sep_moi_fast(sep_et, moi_et):
    idx_start = np.searchsorted(times_full, sep_et - STEP)
    idx_end = np.searchsorted(times_full, moi_et + STEP)
    H = np.zeros(3)
    mass = mass_MCS
    total_ion_used = 0.0
    last_desat = idx_start
    sep_done = False

    for idx in range(idx_start, idx_end):
        t = times_full[idx]

        # Switch inertia at separation
        torque = torque_precompute_MCS[idx]
        if (not sep_done) and t >= sep_et:
            sep_done = True

        if sep_done:
            torque = torque_precompute_MPO[idx]

        H += torque*STEP

        if (t - times_full[last_desat]) >= DT_CRUISE:
            H_norm = np.linalg.norm(H)
            T_avg = H_norm/DT_CRUISE
            F_req = T_avg/(2*ION_GROUP.lever_m)
            fuel = ION_GROUP.fuel_for_duration(F_req, DT_CRUISE)
            total_ion_used += fuel
            mass = max(0.0, mass - fuel)
            H = np.zeros(3)
            last_desat = idx

    total_fuel = total_ion_used + CHEM_BOOSTER_PROP_USED
    return total_fuel

# -------------------- Adaptive sweep --------------------
def adaptive_sweep_fast(sep_start, sep_end, moi_start, moi_end, step_days=1, expansions=2):
    step_sec = step_days*86400
    for exp in range(expansions+1):
        sep_dates = np.arange(sep_start, sep_end+step_sec, step_sec)
        moi_dates = np.arange(moi_start, moi_end+step_sec, step_sec)
        fuel_grid = np.zeros((len(sep_dates), len(moi_dates)))

        for i, sep_et in enumerate(sep_dates):
            for j, moi_et in enumerate(moi_dates):
                if moi_et <= sep_et:
                    fuel_grid[i,j] = np.nan
                    continue
                fuel_grid[i,j] = simulate_sep_moi_fast(sep_et, moi_et)

        min_idx = np.unravel_index(np.nanargmin(fuel_grid), fuel_grid.shape)
        if 0 < min_idx[0] < len(sep_dates)-1 and 0 < min_idx[1] < len(moi_dates)-1:
            # Minimum inside grid
            break
        # Expand edges if minimum on edge
        sep_start -= step_sec
        sep_end += step_sec
        moi_start -= step_sec
        moi_end += step_sec

    opt_sep = sep_dates[min_idx[0]]
    opt_moi = moi_dates[min_idx[1]]
    min_fuel = fuel_grid[min_idx]
    return fuel_grid, sep_dates, moi_dates, opt_sep, opt_moi, min_fuel

# -------------------- Run fast adaptive sweep --------------------
fuel_grid, sep_dates, moi_dates, opt_sep, opt_moi, min_fuel = adaptive_sweep_fast(
    spice.str2et("2026-09-25"),
    spice.str2et("2026-10-05"),
    spice.str2et("2026-11-25"),
    spice.str2et("2026-12-05"),
    step_days=1,
    expansions=2
)

# -------------------- Plot --------------------
def et_to_datetime(et):
    utc = spice.et2utc(et,'ISOC',0)
    return datetime.strptime(utc.split('T')[0], "%Y-%m-%d")

x_dates = [et_to_datetime(t) for t in moi_dates]
y_dates = [et_to_datetime(t) for t in sep_dates]

plt.figure(figsize=(12,6))
plt.imshow(fuel_grid, origin='lower', aspect='auto',
           extent=[mdates.date2num(x_dates[0]), mdates.date2num(x_dates[-1]),
                   mdates.date2num(y_dates[0]), mdates.date2num(y_dates[-1])],
           cmap='viridis')
plt.colorbar(label='Total Fuel Used [kg]')
plt.xlabel('MOI Date')
plt.ylabel('Separation Date')
plt.title('Fuel Usage vs Separation & MOI Dates (Fast Adaptive Sweep)')
plt.gca().xaxis_date()
plt.gca().yaxis_date()
plt.gcf().autofmt_xdate()
plt.show()

# -------------------- Print optimal --------------------
print("Optimal separation ET:", spice.et2utc(opt_sep,'C',0))
print("Optimal MOI ET:", spice.et2utc(opt_moi,'C',0))
print("Minimum total fuel [kg]:", min_fuel)

spice.kclear()
