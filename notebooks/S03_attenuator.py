import numpy as np
from scipy.integrate import solve_ivp

from S01_pulser import pulser1, pulser2
from S02_transmission_line import Td

Rd = 10e3   # Ohms
Ld = 5e-9   # Henries


def attenuator_current(t_array, v_node_func, R, L):
    def rhs(t, i):
        V_node = v_node_func(np.array([t]))[0]
        return (V_node - R * i[0]) / L

    sol = solve_ivp(rhs, (t_array[0], t_array[-1]), y0=[0.0],
                     t_eval=t_array, max_step=1e-11)
    return sol.y[0]


def attenuator1_current(t_array):
    return attenuator_current(t_array, lambda t: pulser1(t - Td), Rd, Ld)


def attenuator2_current(t_array):
    return attenuator_current(t_array, lambda t: pulser2(t - Td), Rd, Ld)