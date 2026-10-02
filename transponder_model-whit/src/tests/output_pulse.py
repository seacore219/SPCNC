import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from spcnc import transponder 

x_pts = np.linspace(0, 20*1.0e-9, num=100)

f = lambda t: transponder.OutputPulse(t) 
f2 = lambda t: transponder.OutputPulse(t, tau=10.0e-9) 

y = [f(x) for x in x_pts]
y2 = [f2(x) for x in x_pts]
print(y)
print(x_pts)

plt.plot(x_pts,y , label = "pulse tau= 1 ns")
plt.plot(x_pts,y2 , label = "pulse tau=10 ns")
#plt.plot(solution.t/1.0e-9, solution.y[1]/1.0e-6, label = "iloop")
plt.xlabel("time [ns]")
plt.ylabel("current [uA]")
plt.legend()
#
plt.savefig("../results/output_pulse.png", bbox_inches='tight')
plt.savefig("../results/output_pulse.pdf", format="pdf", bbox_inches='tight')
#plt.show()


