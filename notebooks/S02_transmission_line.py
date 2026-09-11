import numpy as np

from S01_pulser import pulser1, pulser2

Td = 5e-9   # T3/T4 propagation delay
Z0 = 50     # T3/T4 characteristic impedance
RP = 50     # R4/R11 matched termination -- equals Z0 by design

def source_current(t, pulser_func, R=RP):
    return pulser_func(t) / R


def farend_voltage(t, pulser_func, Td=Td):
    return pulser_func(t - Td)


def line1_source_current(t):
    return source_current(t, pulser1)


def line2_source_current(t):
    return source_current(t, pulser2)


def line1_farend_voltage(t):
    return farend_voltage(t, pulser1)


def line2_farend_voltage(t):
    return farend_voltage(t, pulser2)

