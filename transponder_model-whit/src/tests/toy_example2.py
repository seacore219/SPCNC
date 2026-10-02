import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
import math
import argparse

parser = argparse.ArgumentParser(
                    prog='spcnc_toy_example',
                    description='prototype spiking network simulation',
                    epilog='')

def LinearPulse(t, t0 = 5.0e-9, width=0.5e-9, rise=1.0e-9, fall=0.5e-9): 
  if (t-t0 < 0) or (t-t0 > width+fall+rise): 
    return 0
  if ((t-t0) > 0) and ((t - t0) < rise):
    # rising
    delta_t = t-t0
    #print(delta_t/rise)
    return delta_t/rise 
  if (t-t0 > width+rise) and (t-t0 <= width+fall+rise):
    # falling
    delta_t = (t0+width+fall+rise)-t
    #print(delta_t/fall)
    return (delta_t/fall) # Ohm 
  else: 
   return 1

def OutputPulse(t, t0=1e-9, tau=1.0e-9, mu0=0.2e-9):
  # tau is the decay of the exponetial
  # t0 is approximately the location of 50% peak on the leading edge
  return (1.0 - 1.0/(math.exp((t - t0)/mu0) + 1.0))*math.exp(- ((t - t0)/tau))

"""
Units: can we default to us? 

There are two input pulses of fixed shape.
This defines the input currents 
"""

parser.add_argument("--Ib1init", help="I_b1 initial value in uA", type=float, default=16 )
parser.add_argument("--Ib2init", help="I_b1 initial value in uA", type=float, default=16 )
parser.add_argument("--I1sw", help="input ntron switching current in uA", type=float, default=20 )
parser.add_argument("--I2sw", help="output ntron switching current in uA", type=float, default=20 )
parser.add_argument("--split", help="current splitting fraction", type=float, default=0.3 )
parser.add_argument("--dt", help="delta t in ns", type=float, default=5 )

args = parser.parse_args()

Inw1_sw = args.I1sw*1.0e-6  # input ntron switching current
Inw2_sw = args.I2sw*1.0e-6  # output ntron switching current

Ib1 = args.Ib1init*1.0e-6      # intial input ntron bias 
Ib2 = args.Ib2init*1.0e-6      # initial output ntron bias

# Transponder circuit paramters
Ls1 = 5 * 1.0e-9  # input ntron shunt inductance
Rs1 = 2           # input ntron shunt resistance
LT  = 140 * 1.0e-9# transponder loop inductance
RT0 = 5           # transponder loop resistance

Lnw   = 10 * 1.0e-9 # ntron channel inductance
Rnorm = 1000        # ntron channel normal state resistance 

recover_time1 = 3.0e-9
# Refractory period
ref_period = 0.003 *1.0e-6
# current in loop goes to zero when state goes to 1.


#  first two columns are inputs   
#                          i1  i2   t1   t2    t3
delay = 1.0e-9* np.array([[  0, 0, 9  , 0],   # i1
                          [  0, 0, 9  , 0],   # i2
                          [  0, 0, 0  , 1 ],   # t1
                          [  0, 0, 0  , 0 ],   # t1
                          ])
connectivity = np.minimum(1,np.ceil(delay))

print(np.shape(delay))
print(len(delay))

N_transponders = len(delay)

# state:  0 is superconducting, 1 is clicking, 2 is recovery
input_states = np.zeros(N_transponders)
output_states = np.zeros(N_transponders)

input_click_times = [[] for _ in range(N_transponders)]
output_click_times = [[] for _ in range(N_transponders)]

#output_click_times[0].append(10.0e-9)
#output_click_times[1].append(16.0e-9)

sensor_pulses = [[1.0e-9],[1.0e-9 + 1.0e-9*args.dt]]


transponder_bias1  = Ib1*np.ones(N_transponders) # uA
transponder_bias2  = Ib2*np.ones(N_transponders) # uA

# most of the output ntron goes into a shunt only this fraction goes to the connected inputs
transponder_current_divider = args.split 

## Resistances emulating  firing of ntrons
transponder_Rnw_input = np.zeros(N_transponders) # input 
transponder_RT_loop   = RT0*np.ones(N_transponders)  # output
transponder_input_gate_current = np.zeros(N_transponders) # input 
transponder_source_current = np.zeros(N_transponders) # input 

init_input_current  = transponder_bias1
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

output_states = np.zeros(N_transponders)
#output_click_times = np.array([0 , 0, 0, 0 ,0]) ## could be array
output_click_times = [[] for _ in range(len(input_states))]


def RHS(t, x):
  # y[0] = di_{nw}/dt
  # y[1] = di_{loop}/dt
  nw_currents   = x[::2]  
  loop_currents = x[1::2]

  eps = 1.0e-11

  for n, clicks in enumerate(sensor_pulses):
    for i_click, sensor_click_t in enumerate(clicks):  
      if( abs(t -sensor_click_t) <= eps ) :
        print(f"n = {n} i_click = {i_click} time {t} vs {sensor_click_t}")
        output_states[n] = 1
        # simple threshold model of ntron firing. 
        # OUTPUT Clicks
        output_click_times[n].append(t)#sensor_click_t) # or t?
        sensor_pulses[n].pop(i_click)

  # Update the output currents for each transponder
  for i_out, output_state in enumerate(output_states):
    if output_states[i_out] == 1:
      # output state is firing
      # get last click time and determine if it is reset
      #print(f"output click: {i_out}, { output_click_times[i_out]}")
      tlast                       = output_click_times[i_out][len(output_click_times[i_out])-1]
      transponder_RT_loop[i_out]  = RT0 + Rnorm*LinearPulse(t,t0=tlast,width=1.0e-9)
      # let the output state reset
      if t-tlast > ref_period:
        output_states[i_out] = 0
        print("resetting output")
      #if i_out == 1 or i_out == 0: 
      #  output_states[i_out] = 1
    if output_states[i_out] == 0:
      transponder_RT_loop[i_out] = RT0
      # output state is ready to fire, should it fire?
      if loop_currents[i_out] + transponder_bias2[i_out] > Inw2_sw: 
        print("output {}: click_t = {}".format(i_out, t))
        output_states[i_out] = 1
        # simple threshold model of ntron firing. 
        # OUTPUT Clicks
        output_click_times[i_out].append(t)

  # update the input currents
  # input
  for i_in, input_state in enumerate(input_states):
    if input_states[i_in] == 0:
      # get all the input pulses
      ntron_gate_current = 0
      transponder_input_gate_current[i_in] = 0
      for j_in in input_connections[i_in]:
        n_split = np.sum(connectivity[j_in])
        b2  = transponder_bias2[j_in]
        amp = b2*transponder_current_divider/n_split
        #print("{:d} input from {:d} biased at {}, output {}/{} = {}".format(i_in, j_in, b2, b2*transponder_current_divider, n_split ,amp))
        for click_t in output_click_times[j_in]:
          click_delay         = input_delays[i_in][j_in]
          click_arrival       = click_t  + click_delay
          ntron_gate_current += amp*OutputPulse(t, t0=click_arrival)
          #print(f"input sum {i_in}: click_t = {click_t}, delay[{j_in}] = {click_delay}  ntron gate {ntron_gate_current}")
      # now for the internals of the transponder 
      transponder_source_current[i_in]     = nw_currents[i_in] + ntron_gate_current
      transponder_input_gate_current[i_in] = ntron_gate_current
      if nw_currents[i_in] + ntron_gate_current > Inw1_sw: 
        # INPUT Clicks
        input_states[i_in] = 1
        input_click_times[i_in].append(t)
        #print("node {:d}: click_t = {}".format(i_in,t))
      # add staying dead for recovery time
      #transponder_Rnw_input[i_in] = 0
    if input_states[i_in] == 1:
      #print(f"input fired at t={t}")
      latest_input_click = input_click_times[i_in][len(input_click_times[i_in])-1]
      if t-latest_input_click > ref_period:
        # reset
        print("input reset")
        input_states[i_in] = 0
        transponder_Rnw_input[i_in] = 0.0
      else: 
        transponder_Rnw_input[i_in] = Rnorm*LinearPulse(t,t0=latest_input_click, width=1.0e-9)
  y = []
  for i_node,loop_cur in enumerate(loop_currents):
    i1  = transponder_bias1[i_node]
    inw = nw_currents[i_node]
    iloop = loop_cur
    Rnw =  transponder_Rnw_input[i_node]
    RT  = transponder_RT_loop[i_node]
    #if i_node == 2:
    #  print(f"{i_node} t={t}, Rnw = {Rnw},  RT = {RT}, in_total = {transponder_source_current[i_in]}")
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

  print("Starting simulation")
  solution = solve_ivp(RHS, (0, 100 * 1.0e-9), transponder_initial_state, t_eval= t_pts, max_step = 1.0e-12)

  fig, axs = plt.subplots(N_transponders)
  #fig.suptitle('Vertically stacked subplots')
  axs[0].plot(solution.t/1.0e-9, solution.y[0]/1.0e-6, label='input 1')
  axs[0].plot(solution.t/1.0e-9, solution.y[1]/1.0e-6, label='loop 1')
  axs[1].plot(solution.t/1.0e-9, solution.y[2]/1.0e-6, label='input 2')
  axs[1].plot(solution.t/1.0e-9, solution.y[3]/1.0e-6, label='loop 2')
  axs[2].plot(solution.t/1.0e-9, solution.y[4]/1.0e-6, label='input 3')
  axs[2].plot(solution.t/1.0e-9, solution.y[5]/1.0e-6, label='loop 3')
  axs[3].plot(solution.t/1.0e-9, solution.y[6]/1.0e-6, label='input 3')
  axs[3].plot(solution.t/1.0e-9, solution.y[7]/1.0e-6, label='loop 3')

  axs[0].legend()
  axs[1].legend()
  axs[2].legend()

  for ax in axs.flat:
    ax.set(xlabel='time [ns]',ylabel='current [uA]')

  # Hide x labels and tick labels for top plots and y ticks for right plots.
  for ax in axs.flat:
    ax.label_outer()
  #plt.legend()
  print(solution.y)
  plt.savefig("../results/toy_example.png", bbox_inches='tight')
  plt.savefig("../results/toy_example.pdf", format="pdf", bbox_inches='tight')

  print(f"solution status {solution.status}")
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
