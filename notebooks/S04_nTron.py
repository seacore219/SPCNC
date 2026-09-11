import numpy as np

Jc = 46e9          # A/m^2
width_g = 9.78e-9  # m
thickness = 20e-9  # m
C = 1               # ntron_2.lib default (not overridden in the .asc)

Isw_g = Jc * width_g * thickness * C


def find_threshold_crossings(t, i_array, Ith):
    """Return the times where |i_array| first crosses Ith, rising and falling.

    Returns (t_rise, t_fall): the time of the first upward crossing and the
    first downward crossing after it, or None for either if it never crosses.
    """
    above = np.abs(i_array) > Ith
    crossings = np.diff(above.astype(int))

    rise_idx = np.where(crossings == 1)[0]
    fall_idx = np.where(crossings == -1)[0]

    t_rise = t[rise_idx[0] + 1] if len(rise_idx) > 0 else None
    t_fall = t[fall_idx[0] + 1] if len(fall_idx) > 0 else None

    return t_rise, t_fall