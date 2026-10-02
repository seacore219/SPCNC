import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys

import argparse
parser = argparse.ArgumentParser(
                    prog='spcnc_toy_example',
                    description='prototype spiking network simulation',
                    epilog='')

from spcnc import transponder 

"""
Units: can we default to us? 

There are two input pulses of fixed shape.
This defines the input currents 
"""

parser.add_argument("--Ib1init", help="I_b1 initial value in uA", type=float, default=100 )
parser.add_argument("--Ib2init", help="I_b1 initial value in uA", type=float, default=100 )
parser.add_argument("--I1sw", help="input ntron switching current in uA", type=float, default=150 )
parser.add_argument("--I2sw", help="output ntron switching current in uA", type=float, default=150 )


args = parser.parse_args()


Inw1_sw = 100.0*1.0e-6  # input ntron switching current
Inw2_sw = 100.0*1.0e-6  # output ntron switching current

Ib1 = args.Ib1init*1.0e-6      # intial input ntron bias 
Ib2 = args.Ib2init*1.0e-6      # initial output ntron bias

# Transponder circuit paramters
Ls1 = 5 * 1.0e-9   # input ntron shunt inductance
Rs1 = 3            # input ntron shunt resistance
LT  = 140 * 1.0e-9 # transponder loop inductance
RT0  = 4           # transponder loop resistance

Lnw   = 10 * 1.0e-9 # ntron channel inductance
Rnorm = 1000        # ntron channel normal state resistance 

recover_time1 = 10.0e-9
# Refractory period
ref_period = 0.005 *1.0e-6
# current in loop goes to zero when state goes to 1.


## row = source node, colun = destination node
#                           i1  i2  t1  t2  t3 ## multiplies the array by 10^-6 to get microseconds 
delay = 1.0e-6* np.array([[  0, 0, 0.1, 0.5, 0   ],   # i1 
                          [  0, 0, 0.3, 0  , 0   ],   # i2 
                          [  0, 0, 0  , 0.1, 0   ],   # t1
                          [  0, 0, 0  , 0  , 1.2 ],   # t2
                          [  0, 0, 0  , 0.0, 0   ]    # t3
                   ])
## ix = input x, tx = transponder x
connectivity = np.minimum(1,np.ceil(delay))

print(np.shape(delay))
print(len(delay))

N_transponders = len(delay)

# state:  0 is superconducting, 1 is clicking, 2 is recovery
input_states = np.zeros(N_transponders)
output_states = np.zeros(N_transponders)
input_click_times = [[] for _ in range(len(input_states))]
input_click_times[0].append(0.1e-6)
input_click_times[1].append(0.12e-6)


transponder_bias1  = np.array([0, 0,   60.0e-6, 70.0e-6, 80.0e-6]) # uA
transponder_bias2  = np.array([20.0e-6, 70.0e-6, 50.0e-6, 50.0e-6, 90.0e-6]) # SNSPD bias  for first two entries

# most of the output ntron goes into a shunt only this fraction goes to the connected inputs
transponder_current_divider = 0.1 

## Resistances emulating  firing of ntrons
transponder_Rnw_input = np.zeros(N_transponders) # input 
transponder_RT_loop   = RT0*np.ones(N_transponders)  # output
init_input_current = transponder_bias1
init_loop_currents  = np.zeros(N_transponders)


transponder_initial_state =  np.empty((2*N_transponders), dtype=transponder_RT_loop.dtype)
transponder_initial_state[0::2] = init_input_current
transponder_initial_state[1::2] = init_loop_currents
#print(f"transponder_initial_state = {transponder_initial_state}")

# This matrix gives the columns
input_connections = [[] for _ in range(len(input_states))]
input_delays = delay.T
for i,arow in enumerate(connectivity.T):
  for j,acol in enumerate(arow):
    if acol == 1:
      input_connections[i].append(j)

print("input connections")
print(input_connections)

ouput_states = np.array([0, 0, 0, 0 ,0])
#ouput_click_times = np.array([0 , 0, 0, 0 ,0]) ## could be array
ouput_click_times = [[] for _ in range(len(input_states))]



def RHS(t, x):
  # y[0] = di_{nw}/dt
  # y[1] = di_{loop}/dt
  nw_currents   = x[::2]  
  loop_currents = x[1::2]

  # Update the output currents for each transponder
  for i_out, _ in enumerate(output_states):
    if output_states[i_out] == 1:
      # output state is firing
      # get last click time and determine if it is  reset
      tlast = ouput_click_times[len(ouput_click_times)-1]
      transponder_RT_loop[i_out]  = RT0 + transponder.LinearPulse(t,t0=tlast)
      # let the output state reset
      if t-tlast > ref_period:
        output_states[i_out] = 0
    if output_states[i_out] == 0:
      transponder_RT_loop[i_out] = RT0
      # output state is ready to fire, should it fire?
      if loop_currents[i_out] + transponder_bias2[i_out] > Inw2_sw: 
        print("output {:d}: click_t = {}".format(i_out,t))
        output_states[i_out] = 1
        # simple threshold model of ntron firing. 
        # OUTPUT Clicks
        ouput_click_times[i_out].append(t)

  # update the input currents
  # input
  for i_in, input_state in enumerate(input_states):
    if input_state == 0:
      # get all the input pulses
      ntron_gate_current = 0
      for j_in in input_connections[i_in]:
        n_split = np.sum(connectivity[j_in])
        b2 = transponder_bias2[j_in]
        amp = b2*transponder_current_divider/n_split
        print("{:d} input from {:d} biased at {}, output {}/{} = {}".format(i_in, j_in, b2, b2*transponder_current_divider, n_split ,amp))
        for click_t in ouput_click_times[j_in]:
          click_delay = input_delays[i_in][j_in]
          click_arrival =  click_t  + click_delay
          print("input sum {:d}: click_t = {}, delay[{:d}] = {}".format(i_in,click_t,j_in,click_delay))
          ntron_gate_current +=  amp*transponder.OutputPulse(t, t0=click_arrival)
      # now for the internals of the transponder 
      if nw_currents[i_in] + ntron_gate_current > Inw1_sw: 
        # INPUT Clicks
        input_states[i_in] = 1
        input_click_times[i_in].append(t)
        print("node {:d}: click_t = {}".format(i_in,t))
      # add staying dead for recovery time
    transponder_Rnw_input[i_in] = 0
    if input_states[i_in] == 1:
        latest_input_click = input_click_times[i_in][-1]
        if t-latest_input_click > ref_period:
            # reset
            input_states[i_in] = 0
            transponder_Rnw_input[i_in] = 0.0
        else: 
            transponder_Rnw_input[i_in] = Rnorm*transponder.LinearPulse(t,t0=latest_input_click)
  y = []
  for i_node,loop_cur in enumerate(loop_currents):
    i1  = transponder_bias1[i_node]
    inw = nw_currents[i_node]
    iloop = loop_cur
    Rnw =  transponder_Rnw_input[i_node]
    RT = transponder_RT_loop[i_node]
    y1 = (-inw*Ls1*Rnw - i1*LT*Rs1 + iloop*LT*Rs1 + inw*LT*(Rnw + Rs1) + iloop*Ls1*RT)/(Lnw*(Ls1 - LT) - Ls1*LT)
    y2 = (-inw*Ls1*Rnw - i1*Lnw*Rs1 + iloop*Lnw*Rs1 + inw*Lnw*Rs1 + iloop*Lnw*RT + iloop*Ls1*RT)/(Lnw*(Ls1 - LT) - Ls1*LT)
    y.append(y1)
    y.append(y2)
  return y



def main() -> int:
  """ Prototype network simulation"""

  print(delay)
  print(connectivity)
  print(connectivity.T)
  
  print("connectivity row sums give the number of loads for each output signal")
  for acol in connectivity:
    print(np.sum(acol))
  # transpose and sum to get number of connections to transponder output.
  print("connectivity column sums give the number of inputs")
  for acol in connectivity.T:
    print(np.sum(acol))


  for acol in delay.T:
    print("col : {}\n".format(acol))


  t_pts = np.linspace(0, 100*1.0e-9, num=1000)

  solution = solve_ivp(RHS, (0, 100 * 1.0e-9), transponder_initial_state, t_eval= t_pts, max_step = 1.0e-11)

  fig, axs = plt.subplots(3)
  #fig.suptitle('Vertically stacked subplots')
  axs[0].plot(solution.t/1.0e-9, solution.y[0]/1.0e-6, label='input 1')
  axs[0].plot(solution.t/1.0e-9, solution.y[1]/1.0e-6, label='loop 1')
  axs[1].plot(solution.t/1.0e-9, solution.y[2]/1.0e-6, label='input 2')
  axs[1].plot(solution.t/1.0e-9, solution.y[3]/1.0e-6, label='loop 2')
  axs[2].plot(solution.t/1.0e-9, solution.y[4]/1.0e-6, label='input 3')
  axs[2].plot(solution.t/1.0e-9, solution.y[5]/1.0e-6, label='loop 3')


  for ax in axs.flat:
    ax.set(xlabel='time [ns]',ylabel='current [uA]')

  # Hide x labels and tick labels for top plots and y ticks for right plots.
  for ax in axs.flat:
    ax.label_outer()
  #
  #plt.legend()
  #
  print(solution.y)
  plt.savefig("../results/toy_example.png", bbox_inches='tight')
  plt.savefig("../results/toy_example.pdf", format="pdf", bbox_inches='tight')
  plt.show()


  return 0

if __name__ == '__main__':
  sys.exit(main())  # next section explains the use of sys.exit




# for each transponder input
# an output click carries a pulse amplitude/shape and delay 
# for long lines multple  current spikes may be present, need dynamic sizing




#f = lambda t: transponder.OutputPulse(t) 
#f2 = lambda t: transponder.OutputPulse(t, tau=10.0e-9) 
#
#y = [f(x) for x in x_pts]
#y2 = [f2(x) for x in x_pts]
#print(y)
#print(x_pts)
#
#
#plt.plot(solution.t/1.0e-9, solution.y[0]/1.0e-6, label = "inw")
#plt.plot(solution.t/1.0e-9, solution.y[1]/1.0e-6, label = "iloop")
#
#plt.plot(x_pts,y , label = "pulse tau= 1 ns")
#plt.plot(x_pts,y2 , label = "pulse tau=10 ns")
##plt.plot(solution.t/1.0e-9, solution.y[1]/1.0e-6, label = "iloop")
#plt.xlabel("time [ns]")
#plt.ylabel("current [uA]")
#plt.legend()
##
#plt.savefig("../results/output_pulse.png", bbox_inches='tight')
#plt.savefig("../results/output_pulse.pdf", format="pdf", bbox_inches='tight')
##plt.show()
#
#
#
