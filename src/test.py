import os

kernels = [
    "naif_spice/cassini_kernels/lsk/latest_leapseconds.tls",
    "naif_spice/cassini_kernels/fk/cas_v37.tf",
    "naif_spice/cassini_kernels/pck/020514_SE_SAT105.bsp",
    "naif_spice/cassini_kernels/pck/981005_PLTEPH-DE405S.bsp",
    "naif_spice/cassini_kernels/pck/cpck05Mar2004.tpc",
    "naif_spice/cassini_kernels/sclk/cas0084.tsc",
    "naif_spice/cassini_kernels/sclk/cas_iss_v09.ti",
    "naif_spice/cassini_kernels/sclk/04135_04171pc_psiv2.bc",
    "naif_spice/cassini_kernels/sclk/030201AP_SK_SM546_T45.bsp"
]

for k in kernels:
    print(k, "exists?" , os.path.exists(k))
    
print("Current working directory:", os.getcwd())