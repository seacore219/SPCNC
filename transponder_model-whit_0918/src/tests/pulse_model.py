import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from spcnc import transponder 

x_pts = np.linspace(0, 10*1.0e-9, num=100)

f = lambda t: transponder.LinearPulse(t,5.0e-9, 1.0e-9, 3.0e-9, 3.0e-9) 

y = [f(x) for x in x_pts]
print(y)
print(x_pts)

plt.plot(x_pts,y , label = "pulse")
#plt.plot(solution.t/1.0e-9, solution.y[1]/1.0e-6, label = "iloop")
plt.xlabel("time [ns]")
plt.ylabel("current [uA]")
plt.legend()
#
plt.savefig("../results/pulse_model.png", bbox_inches='tight')
plt.savefig("../results/pulse_model.pdf", format="pdf", bbox_inches='tight')
#plt.show()


