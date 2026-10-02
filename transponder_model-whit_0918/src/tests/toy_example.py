import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
import math
import argparse
import spcnc.tof_input_generator as tof_gen

parser = argparse.ArgumentParser( # parses command line arguments
                    prog='spcnc_toy_example',
                    description='prototype spiking network simulation',
                    epilog='')

def LinearPulse(t, t0 = 5.0e-9, width=0.5e-9, rise=1.0e-9, fall=0.5e-9): ## makes the trapezoid pulse
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

def OutputPulse(t, t0=1e-9, tau=1.0e-9, mu0=0.2e-9): ## sigmoid from 0 to 1 as t goes to t0 for the output rise
  ## tau is the decay of the exponetial
  ## t0 is approximately the location of 50% peak on the leading edge
  return (1.0 - 1.0/(math.exp((t - t0)/mu0) + 1.0))*math.exp(- ((t - t0)/tau)) # exponential decay

"""
Units: can we default to us? 

There are two input pulses of fixed shape.
This defines the input currents 
"""
# flags
parser.add_argument("--output", help="output plots file name", default="toy_example" )
parser.add_argument("--Ib1init", help="I_b1 initial value in uA", type=float, default=16 )
parser.add_argument("--Ib2init", help="I_b1 initial value in uA", type=float, default=16 )
parser.add_argument("--I1sw", help="input ntron switching current in uA", type=float, default=20 )
parser.add_argument("--I2sw", help="output ntron switching current in uA", type=float, default=20 )
parser.add_argument("--split", help="current splitting fraction", type=float, default=0.1 )
parser.add_argument("--generator", help="use tof generator" , action='store_true' )
parser.add_argument("--dt", help="delta t in ns, ignored if generator is used", type=float, default=5 )

parser.add_argument('--ib1', nargs=2, action='append', type=float, help='set node [0] input bias to value [1]')
parser.add_argument('--ib2', nargs=2, action='append', type=float, help='set node [0] output bias to value [1]')
## nargs = 2 means expects exactly 2 arguments
parser.add_argument('--trace_node', type=int, default=4, help='matrix index of node for the gate-current trace plot (0=i1, 1=i2, 2=t1 ... 7=t6); default t3')
args = parser.parse_args()

print(args.ib1)

Inw1_sw = args.I1sw*1.0e-6  ## input ntron switching current ; takes in raw command line values in uA, converts to uA by doing x1e-6 
Inw2_sw = args.I2sw*1.0e-6  ## output ntron switching current

Ib1 = args.Ib1init*1.0e-6      ## intial input ntron bias 
Ib2 = args.Ib2init*1.0e-6      ## initial output ntron bias

# Transponder circuit paramters
Ls1 = 5 * 1.0e-9  ## input ntron shunt inductance
Rs1 = 2           ## input ntron shunt resistance
LT  = 150 * 1.0e-9## transponder loop inductance
RT0 = 3           ## transponder loop resistance

Lnw   = 10 * 1.0e-9 ## ntron channel inductance
Rnorm = 1000        ## ntron channel normal state resistance 

recover_time1 = 3.0e-9 ## not actuall used??
# Refractory period
ref_period = 0.003 *1.0e-6 ## also not used??
# current in loop goes to zero when state goes to 1.


#  first two columns are inputs   
##                       to: i1  i2  t1  t2  t3  t4  t5  t6    from:
delay = 1.0e-9* np.array([[  0 , 0 , 5 , 0 , 0 , 0 , 9 , 0],   # i1
                          [  0 , 0 , 0 , 5 , 0 , 0 , 0 , 5],   # i2
                          [  0 , 0 , 0 , 5 , 5 , 0 , 0 , 0],   # t1  -> t2, t3          (recurrent block 1-2-3)
                          [  0 , 0 , 5 , 0 , 5 , 0 , 0 , 0],   # t2  -> t1, t3          (recurrent block 1-2-3)
                          [  0 , 0 , 5 , 5 , 0 , 5 , 0 , 0],   # t3  -> t1, t2 ; t4     (t3->t4 is the ONLY feedforward link)
                          [  0 , 0 , 0 , 0 , 0 , 0 , 5 , 5],   # t4  -> t5, t6          (recurrent block 4-5-6)
                          [  0 , 0 , 0 , 0 , 0 , 5 , 0 , 5],   # t5  -> t4, t6          (recurrent block 4-5-6)
                          [  0 , 0 , 0 , 0 , 0 , 5 , 5 , 0],   # t6  -> t4, t5          (recurrent block 4-5-6)
                          ])
## i1->t1 (5ns), i1->t5 (9ns), i2->t2 (5ns), i2->t6 (5ns) sensor inputs (unchanged)
## t1<->t2<->t3<->t1 all-to-all recurrent, t3->t4 feedforward (no t4->t3), t4<->t5<->t6<->t4 all-to-all recurrent
## a nonzero entry = connection with that delay in ns; set an entry to 0 to cut that connection
connectivity = np.minimum(1,np.ceil(delay)) # converts binarizes delay function to either 0 or 1 (connnection or no connection)

print(np.shape(delay)) ## prints (8,8) for the size of the delay matrix
print(len(delay)) ## prints the number of transponders

N_transponders = len(delay) ## number of transponders is number of rows

# state:  0 is superconducting, 1 is clicking, 2 is recovery
input_states = np.zeros(N_transponders) ## initialized as 0s since nothing fired yet, no current through
output_states = np.zeros(N_transponders) ## initialized as 0s since nothing fired yet, no current through

input_click_times = [[] for _ in range(N_transponders)] ## history of click (firing) times [0,0,0,0,0,0,0,0]
output_click_times = [[] for _ in range(N_transponders)] ## history of click (firing) times [0,0,0,0,0,0,0,0]

dt = 1.0e-9*args.dt
if args.generator:
  print("using generator")
  ev = tof_gen.generate_tof_event()
  dt = ev["tof"]

sensor_pulses = [[1.0e-9],[1.0e-9 + dt]] ## when the pulsers fire; 1st at 1ns, 2nd at given time by args or 5ns as default

transponder_bias1  = Ib1*np.ones(N_transponders) # uA ## skipped if no arg given; uniform biases in that case [1,1,1,1,1,1,1,1], multiplied by Ib1=16e-6 when no custom bias given
transponder_bias2  = Ib2*np.ones(N_transponders) # uA ## skipped if no arg given; uniform biases in that case [1,1,1,1,1,1,1,1], multiplied by Ib2=16e-6 when no custom bias given

if args.ib1: # if 
  for setting in args.ib1:
    #print(f"setting node {setting[0]} b1 = {setting[1]}")
    transponder_bias1[int(setting[0])] = setting[1]*1.0e-6 # if used, converts to uA
if args.ib2:
  for setting in args.ib2:
    transponder_bias2[int(setting[0])] = setting[1] *1.0e-6

# most of the output ntron goes into a shunt only this fraction goes to the connected inputs
transponder_current_divider = args.split # split is the fraction of the output current that goes to the connected inputs. The rest goes to the shunt.

## Resistances emulating firing of ntrons; this is updated every time step by RHS
transponder_Rnw_input = np.zeros(N_transponders) # input 
transponder_RT_loop   = RT0*np.ones(N_transponders)  # output
transponder_input_gate_current = np.zeros(N_transponders) # input 
transponder_source_current = np.zeros(N_transponders) # input 

init_input_current  = transponder_bias1 # starting input currents = bias currents
init_loop_currents  = np.zeros(N_transponders)


transponder_initial_state =  np.empty((2*N_transponders), dtype=transponder_RT_loop.dtype) #initializes 16 slots for current values
transponder_initial_state[0::2] = init_input_current
transponder_initial_state[1::2] = init_loop_currents
#print(f"transponder_initial_state = {transponder_initial_state}")

# This matrix gives the columns
input_connections = [[] for _ in range(len(input_states))] 
input_delays = delay.T ## .T transposes the matrix 
for i,arow in enumerate(connectivity.T):
  for j,acol in enumerate(arow):
    if acol == 1:
      input_connections[i].append(j)

#print("input connections:")
#print(input_connections)

output_states = np.zeros(N_transponders) ## initializes the outputs to 0 since no firing yet
#output_click_times = np.array([0 , 0, 0, 0 ,0]) ## could be array
output_click_times = [[] for _ in range(len(input_states))] ## 


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
      ## Simple nTron firing as a change in resistance
      transponder_RT_loop[i_out]  = RT0 + Rnorm*LinearPulse(t,t0=tlast,width=2.0e-9)
      # let the output state reset
      if t-tlast > ref_period:
        output_states[i_out] = 0
        print("resetting output ntron")
    if output_states[i_out] == 0:
      transponder_RT_loop[i_out] = RT0
      # output state is ready to fire, should it fire?
      if loop_currents[i_out] + transponder_bias2[i_out] > Inw2_sw: 
        # simple threshold model of ntron firing. 
        print("output {}: click_t = {}".format(i_out, t))
        output_states[i_out] = 1
        output_click_times[i_out].append(t)

  # update the input currents
  # input
  for i_in, input_state in enumerate(input_states):
    if input_states[i_in] == 0:
      transponder_Rnw_input[i_in] = 0
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
    if input_states[i_in] == 1:
      #print(f"input fired at t={t}")
      latest_input_click = input_click_times[i_in][len(input_click_times[i_in])-1]
      if t-latest_input_click > ref_period:
        # reset
        print("input reset")
        input_states[i_in] = 0
        #transponder_Rnw_input[i_in] = 0.0
      else: 
        # Turn on the loop resistance (effectivly the ntron gate-source resistance when firing)
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
  #print(connectivity)
  #print(connectivity.T)
  
  #print("connectivity row sums give the number of loads for each output signal")
  #for acol in connectivity:
  #  print(np.sum(acol))
  ## transpose and sum to get number of connections to transponder output.
  #print("connectivity column sums give the number of inputs")
  #for acol in connectivity.T:
  #  print(np.sum(acol))

  #for acol in delay.T:
  #  print("col : {}\n".format(acol))

  t_pts = np.linspace(0, 100*1.0e-9, num=1000)

  print(f"shunt/output current splilt: {transponder_current_divider}")

  print("Starting simulation")
  solution = solve_ivp(RHS, (0, 100 * 1.0e-9), transponder_initial_state, t_eval= t_pts, max_step = 1.0e-12)

  fig, axs = plt.subplots(6)
  #fig.suptitle('Vertically stacked subplots')
  axs[0].plot(solution.t/1.0e-9, solution.y[4]/1.0e-6, label='gate 2')
  axs[1].plot(solution.t/1.0e-9, solution.y[5]/1.0e-6, 'r',label='loop 2')
  axs[2].plot(solution.t/1.0e-9, solution.y[6]/1.0e-6, label='gate 3')
  axs[3].plot(solution.t/1.0e-9, solution.y[7]/1.0e-6, 'r',label='loop 3')
  axs[4].plot(solution.t/1.0e-9, solution.y[8]/1.0e-6, label='gate 4')
  axs[5].plot(solution.t/1.0e-9, solution.y[9]/1.0e-6, 'r',label='loop 4')
  #axs[2].plot(solution.t/1.0e-9, solution.y[4]/1.0e-6, label='input 3')
  #axs[2].plot(solution.t/1.0e-9, solution.y[5]/1.0e-6, label='loop 3')
  #axs[3].plot(solution.t/1.0e-9, solution.y[6]/1.0e-6, label='input 3')
  #axs[3].plot(solution.t/1.0e-9, solution.y[7]/1.0e-6, label='loop 3')

  axs[0].annotate(f"dt={dt//1.0e-9}, \nb1={transponder_bias1/1.0e-6}, \nb2={transponder_bias2/1.0e-6}", xycoords="subfigure points",xy=(1,1))

  axs[0].legend()
  axs[1].legend()
  axs[2].legend()
  axs[3].legend()
  axs[4].legend()
  axs[5].legend()

  for ax in axs.flat:
    ax.set(xlabel='time [ns]',ylabel='current [uA]')

  # Hide x labels and tick labels for top plots and y ticks for right plots.
  for ax in axs.flat:
    ax.label_outer()
  #plt.legend()
  plt.savefig(f"../results/{args.output}.png", bbox_inches='tight',dpi=200)
  plt.savefig(f"../results/{args.output}.pdf", format="pdf", bbox_inches='tight',dpi=200)

  #print(solution.y)
  print(f"solution status {solution.status}")
  #plt.show()

  # ------------------------------------------------------------------
  # Post-processing: rebuild the gate current that RHS computes but throws away.
  # Same formula as the input block of RHS: each upstream click at click_t arrives
  # at click_t + delay and adds amp*OutputPulse. Uses the click times RHS recorded.
  # ------------------------------------------------------------------
  t   = solution.t
  tns = t/1.0e-9
  names = ['i1', 'i2'] + [f't{k}' for k in range(1, N_transponders-1)]
  gate_parts = [dict() for _ in range(N_transponders)]   # gate_parts[i][j] = current into i from j
  gate_current = np.zeros((N_transponders, len(t)))
  for i_in in range(N_transponders):
    for j_in in input_connections[i_in]:
      amp  = transponder_bias2[j_in]*transponder_current_divider/np.sum(connectivity[j_in])
      part = np.zeros(len(t))
      for click_t in output_click_times[j_in]:
        arrival = click_t + input_delays[i_in][j_in]
        part += np.array([amp*OutputPulse(tk, t0=arrival) if tk >= click_t else 0.0 for tk in t])
      gate_parts[i_in][j_in] = part
      gate_current[i_in] += part

  print("output (spike) click times [ns]:")
  for n in range(N_transponders):
    print(f"  {names[n]}: {np.round(np.array(output_click_times[n])/1.0e-9, 2)}")

  # ---- 1) spike raster ----
  fig, ax = plt.subplots(figsize=(8, 3.5))
  for n in range(N_transponders):
    ax.vlines(np.array(output_click_times[n])/1.0e-9, n-0.4, n+0.4, color='k', lw=2,
              label='output click (spike)' if n == 0 else None)
    if input_click_times[n]:
      ax.plot(np.array(input_click_times[n])/1.0e-9, [n-0.45]*len(input_click_times[n]), 'v', color='tab:red', ms=4,
              label='input (gate) click' if not any(input_click_times[:n]) else None)
  for y in (1.5, 4.5):   # separate sensors | block 1-2-3 | block 4-5-6
    ax.axhline(y, color='0.8', ls='--', lw=0.8)
  ax.set_yticks(range(N_transponders), names)
  ax.set_ylim(N_transponders-0.5, -0.5)
  ax.set(xlim=(0, tns[-1]), xlabel='time [ns]', title='spike raster')
  ax.legend(loc='upper right', fontsize=8, frameon=False)
  plt.savefig(f"../results/{args.output}_raster.png", bbox_inches='tight', dpi=200)
  plt.savefig(f"../results/{args.output}_raster.pdf", bbox_inches='tight')

  # ---- 2) gate-current trace for one node: pulses summing up to threshold ----
  n = args.trace_node
  inw = solution.y[2*n]
  fig, axs2 = plt.subplots(2, sharex=True, figsize=(8, 5))
  for j_in, part in gate_parts[n].items():
    axs2[0].plot(tns, part/1.0e-6, '--', label=f'from {names[j_in]}')
  axs2[0].plot(tns, gate_current[n]/1.0e-6, 'k', lw=1.5, label='summed gate current')
  axs2[0].axhline((Inw1_sw - transponder_bias1[n])/1.0e-6, color='r', ls=':', label='needed to switch (I1sw - Ib1)')
  axs2[0].set(ylabel='gate current [uA]', title=f'{names[n]}: gate current')
  axs2[0].legend(fontsize=8, frameon=False)
  axs2[1].plot(tns, inw/1.0e-6, color='0.5', label='channel current i_nw')
  axs2[1].plot(tns, (inw + gate_current[n])/1.0e-6, 'k', label='i_nw + gate')
  axs2[1].axhline(Inw1_sw/1.0e-6, color='r', ls=':', label='I1sw (threshold)')
  for k, tc in enumerate(input_click_times[n]):
    axs2[1].axvline(tc/1.0e-9, color='r', alpha=0.4, label='input click' if k == 0 else None)
  axs2[1].set(xlabel='time [ns]', ylabel='current [uA]')
  axs2[1].legend(fontsize=8, frameon=False)
  plt.savefig(f"../results/{args.output}_gate_trace_{names[n]}.png", bbox_inches='tight', dpi=200)
  plt.savefig(f"../results/{args.output}_gate_trace_{names[n]}.pdf", bbox_inches='tight')

  # ---- 3) gradient raster: white = no current, black = max current, back to white after a click ----
  # shade = gate current / largest gate current anywhere in the run (pitch black = maximum current)
  gmax  = gate_current.max() if gate_current.max() > 0 else 1.0
  shade = gate_current/gmax
  for n in range(N_transponders):
    for tc in input_click_times[n]:            # white while the nanowire is switched/recovering
      shade[n][(t > tc) & (t <= tc + ref_period)] = 0
  fig, ax = plt.subplots(figsize=(8, 3.5))
  im = ax.imshow(shade, aspect='auto', cmap='Greys', vmin=0, vmax=1, interpolation='nearest',
                 extent=(tns[0], tns[-1], N_transponders-0.5, -0.5))
  for n in range(N_transponders):              # clicks as red lines, so they stand apart from the grey current
    ax.vlines(np.array(output_click_times[n])/1.0e-9, n-0.5, n+0.5, color='red', lw=1.8,
              label='output click (spike)' if n == 0 else None)
    ax.vlines(np.array(input_click_times[n])/1.0e-9, n-0.5, n+0.5, color='red', lw=0.8, ls=':',
              label='input (gate) click' if n == 2 else None)
  ax.legend(loc='upper center', fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.15))
  for y in (1.5, 4.5):
    ax.axhline(y, color='0.8', ls='--', lw=0.8)
  ax.set_yticks(range(N_transponders), names)
  ax.set(xlabel='time [ns]', title='gradient raster: gate current (white = 0, black = max)')
  fig.colorbar(im, ax=ax, label=f'gate current / {gmax/1.0e-6:.2f} uA', pad=0.01)
  plt.savefig(f"../results/{args.output}_gradient_raster.png", bbox_inches='tight', dpi=200)
  plt.savefig(f"../results/{args.output}_gradient_raster.pdf", bbox_inches='tight')

  return 0

if __name__ == '__main__':
  sys.exit(main())  # next section explains the use of sys.exit
