import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math

def OutputPulse(t, t0=1e-9, tau=1.0e-9, mu0=0.2e-9):
  # tau is the decay of the exponetial
  # t0 is approximately the location of 50% peak on the leading edge
  return (1.0 - 1.0/(math.exp((t - t0)/mu0) + 1.0))*math.exp(- ((t - t0)/tau))

def LinearPulse(t, t0 = 5.0e-9, width=1.0e-9, rise=1.0e-9, fall=1.0e-9): 
  if (t-t0 < -rise) or (t-t0 > width+fall): 
    return 0
  if ((t-t0) > -rise) and ((t - t0) < 0):
    # rising
    delta_t = t-t0+rise
    print(delta_t/rise)
    return delta_t/rise 
  if (t-t0 > width) and (t-t0 <= width+fall):
    # falling
    delta_t = (t0+width+fall)-t
    print(delta_t/fall)
    return (delta_t/fall) # Ohm 
  else: 
   return 1

class DelayConnect:
  def __init__(self,src,destination):
    self.src=src
    self.destination = destination

class Transponder:
  def __init__(self):
    self.Inw1_sw = 20.0 * 1.0e-6
    self.Inw2_sw = 20.0 * 1.0e-6
    self.Ib1 = 10.0 * 1.0e-6
    self.Ib2 = 20.0 * 1.0e-6
    self.Ls1 = 5 * 1.0e-9 
    self.Rs1 = 3
    self.LT  = 140 * 1.0e-9
    self.RT  = 4
    self.Lnw = 10 * 1.0e-9
    self.Rnorm = 1000
    self.Ithresh1 = self.Inw1_sw - self.Ib1
    self.Ithresh2 = self.Inw2_sw - self.Ib2
    self.recover_time1 = 3000;
    # Refractory period
    self.ref_period = 5*1.0e-9
    # state:  0 is superconducting, 1 is clicking, 2 is recovery
    self.state = 0
    # t0 time since last click
    self.t0    = 0
    # current in loop goes to zero when state goes to 1.
    self.Iloop = 0

  #def RHS(self,t, x):
  #  # y[0] = di_{nw}/dt
  #  # y[1] = di_{loop}/dt
  #  inw, iloop = x
  #  if inw + Iinput > self.Ithresh1: 
  #    Rnw = Rnorm*LinearPulse(t)
  #    # add staying dead for recovery time
  #  else:
  #    Rnw = 0
  #  Ls1 = self.Ls1
  #  Rs1 = self.Rs1
  #  LT  = self.LT 
  #  if iloop > self.Ithresh2: 
  #    RT = Rnorm2*LinearPulse(t)
  #    # 
  #  else:
  #    RT  = self.RT 
  #  Lnw = self.Lnw
  #  y = [  (-inw *Ls1 *Rnw - i1*LT*Rs1 + iloop*LT*Rs1 + inw*LT*(Rnw + Rs1) + iloop*Ls1*RT)/(Lnw*(Ls1 - LT) - Ls1*LT), 
  #       (-inw*Ls1*Rnw - i1*Lnw*Rs1 + iloop*Lnw*Rs1 + inw*Lnw*Rs1 + iloop*Lnw*RT + iloop*Ls1*RT)/(Lnw*(Ls1 - LT) - Ls1*LT)]
  #  return y

def Rnanowire(t, t0 = 5.0e-9, width=1.0e-9, rise=1.0e-9, fall=1.0e-9, R=1000): 
  # pulse with linear rise/fall time
  if (t < t0) or (t > t0+width+rise+fall): 
    return 0
  if (t > t0) and (t <= t0+rise):
    # rising
    delta_t = t-t0
    return (delta_t/rise)*R # Ohm 
  if (t > t0+rise+width) and (t <= t0+rise+width+fall):
    delta_t = (t0+rise+width+fall)-t
    return (delta_t/fall)*R # Ohm 
  else: 
   return R


