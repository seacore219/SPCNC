import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

Ls1 = 5 * 1.0e-9# nH
Rs1 = 3 # Ohm 
LT = 140  * 1.0e-9# nH
RT = 4  # Ohm 
Lnw = 10 * 1.0e-9# nH ? 
Rnwnorm = 1000
ibias = 10.0 * 1.0e-6

def Rnanowire(t, t0 = 5.0e-9, width=1.0e-9, rise=1.0e-9, fall=1.0e-9): 
  if (t < t0) or (t > t0+width+rise+fall): 
    return 0
  if (t > t0) and (t <= t0+rise):
    # rising
    delta_t = t-t0
    return (delta_t/rise)*Rnwnorm # Ohm 
  if (t > t0+rise+width) and (t <= t0+rise+width+fall):
    delta_t = (t0+rise+width+fall)-t
    return (delta_t/fall)*Rnwnorm # Ohm 
  else: 
   return Rnwnorm # Ohm 

i1 = ibias  

def RHS(t, x):
  inw, iloop = x
  # Multiple pulses
  Rnw = Rnanowire(t) + Rnanowire(t,t0=20.0e-9) + Rnanowire(t,t0=45.0e-9)
  y = [  (-inw *Ls1 *Rnw - i1*LT*Rs1 + iloop*LT*Rs1 + inw*LT*(Rnw + Rs1) + iloop*Ls1*RT)/(Lnw*(Ls1 - LT) - Ls1*LT), 
       (-inw*Ls1*Rnw - i1*Lnw*Rs1 + iloop*Lnw*Rs1 + inw*Lnw*Rs1 + iloop*Lnw*RT + iloop*Ls1*RT)/(Lnw*(Ls1 - LT) - Ls1*LT)]
  return y

x_pts = np.linspace(0, 100*1.0e-9, 1000)

solution = solve_ivp(RHS, (0, 100 * 1.0e-9), [ibias, 0], t_eval= x_pts, max_step = 1.0e-11)

plt.plot(solution.t/1.0e-9, solution.y[0]/1.0e-6, label = "inw")
plt.plot(solution.t/1.0e-9, solution.y[1]/1.0e-6, label = "iloop")
plt.xlabel("time [ns]")
plt.ylabel("current [uA]")
plt.legend()
#
print(solution.y)
plt.savefig("../results/current_loop.png", bbox_inches='tight')
plt.savefig("../results/current_loop.pdf", format="pdf", bbox_inches='tight')
#plt.show()


