import numpy as np
 
def pulse(t, TD, TR, PW, TF, V0, Vmax):

    t1 = TD
    t2 = TD + TR
    t3 = TD + TR + PW
    t4 = TD + TR + PW + TF
 
    return np.piecewise(
        t,
        [t < t1,
         (t >= t1) & (t < t2),
         (t >= t2) & (t < t3),
         (t >= t3) & (t < t4),
         t >= t4],
        [V0,
         lambda t: V0 + (Vmax - V0) * (t - t1) / TR,
         Vmax,
         lambda t: Vmax - (Vmax - V0) * (t - t3) / TF,
         V0]
    )
 
 
# Channel 1 (V1 source in the netlist)
TD1, TR1, PW1, TF1, PER1, NP1 = 16.5e-9, 2.9e-9, 0.1e-9, 2.9e-9, 10e-9, 1
# Channel 2 (V4 source in the netlist)
TD2, TR2, PW2, TF2, PER2, NP2 = 1e-9, 2.9e-9, 0.1e-9, 2.9e-9, 10e-9, 1
 
V1_level, V2_level = 0., 0.25
 
 
def pulser1(t):
    return pulse(t, TD1, TR1, PW1, TF1, V1_level, V2_level)
 
 
def pulser2(t):
    return pulse(t, TD2, TR2, PW2, TF2, V1_level, V2_level)