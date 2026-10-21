# ======================================================================================================
# COMMENTED COPY of toy_example.py  (same code, comments only; the code is checked to be identical)
#
# Comment key
#   [orig l.N]  line number in toy_example.py on `main` (Whit's original, 317 lines)
#   [new]       code that is not in the original (your requests: recurrent wiring, raster, gate trace,
#               gradient raster)
#   schematic:  the part of single_transponder_simple.pdf the line corresponds to
#               U3 = input nTron (Q1), U2 = output nTron (Q2), bias3 = I_b1, bias4 = I_b2,
#               loop 1 = input/gate loop, loop 2 = Q1 shunt (L1, R3), loop 3 = storage (Lb1, R2),
#               loop 4 = output (L2, R1)
#
# How the program runs: (1) read flags, (2) set up the network tables and starting currents,
# (3) RHS() says how fast every current changes at an instant, (4) main() hands RHS to a solver that
# steps through 100 ns, (5) plots.
# ======================================================================================================

# ---------------- STEP 1: libraries ----------------
# [orig l.1] arrays: the network tables and the current vectors
import numpy as np
# [orig l.2] the solver that steps the circuit equations forward in time
from scipy.integrate import solve_ivp
# [orig l.3] plotting
import matplotlib.pyplot as plt
# [orig l.4] used at the very end to return the exit status
import sys
# [orig l.5] exp() for OutputPulse
import math
# [orig l.6] reads the command-line flags
import argparse
# [orig l.7] time-of-flight generator, only used with --generator
import spcnc.tof_input_generator as tof_gen

# [orig l.9] create the flag reader
parser = argparse.ArgumentParser(
                    prog='spcnc_toy_example',
                    description='prototype spiking network simulation',
                    epilog='')


# ---------------- STEP 2: the two pulse shapes ----------------
# [orig l.14] LinearPulse: trapezoid between 0 and 1. x Rnorm = an nTron's resistance while it is switched.
#   t = now, t0 = start, width = flat top, rise / fall = ramp times
def LinearPulse(t, t0 = 5.0e-9, width=0.5e-9, rise=1.0e-9, fall=0.5e-9):
  # [orig l.15-16] before the pulse starts or after it ends: 0 (nTron superconducting)
  if (t-t0 < 0) or (t-t0 > width+fall+rise): 
    return 0
  # [orig l.17-21] ramping up: straight line 0 -> 1 over `rise`
  if ((t-t0) > 0) and ((t - t0) < rise):
    delta_t = t-t0
    return delta_t/rise 
  # [orig l.22-26] ramping down: straight line 1 -> 0 over `fall`
  if (t-t0 > width+rise) and (t-t0 <= width+fall+rise):
    delta_t = (t0+width+fall+rise)-t
    return (delta_t/fall)
  # [orig l.27-28] flat top: 1
  else: 
   return 1

# [orig l.30] OutputPulse: the current one upstream spike delivers into a node's gate (loop 1).
#   t0 = where it is about half way up, tau = decay time, mu0 = how sharp the rise is
def OutputPulse(t, t0=1e-9, tau=1.0e-9, mu0=0.2e-9):
  # [orig l.33] (smooth step up around t0) x (exponential decay); peak about 0.61. Note it is already ~20% of nominal 0.35 ns BEFORE t0
  return (1.0 - 1.0/(math.exp((t - t0)/mu0) + 1.0))*math.exp(- ((t - t0)/tau))

"""
Units: can we default to us? 

There are two input pulses of fixed shape.
This defines the input currents 
"""

# ---------------- STEP 3: the command-line flags (the knobs) ----------------
# [orig l.41] each flag below becomes a variable. Biases / thresholds are in microamps here.
# [orig l.42] base name of the plot files written to ../results
parser.add_argument("--output", help="output plots file name", default="toy_example" )
# [orig l.43] input bias of every node (uA) | [schematic] I_b1, bias3 -> U3 drain (LTspice ~150 uA) | default 16
parser.add_argument("--Ib1init", help="I_b1 initial value in uA", type=float, default=16 )
# [orig l.44] output bias of every node (uA) | [schematic] I_b2, bias4 -> U2 drain (LTspice ~160 uA) | default 16
parser.add_argument("--Ib2init", help="I_b1 initial value in uA", type=float, default=16 )
# [orig l.45] U3 switching current = input threshold (uA) | [schematic] U3 | default 20
parser.add_argument("--I1sw", help="input ntron switching current in uA", type=float, default=20 )
# [orig l.46] U2 switching current = output threshold (uA) | [schematic] U2 | default 20
parser.add_argument("--I2sw", help="output ntron switching current in uA", type=float, default=20 )
# [orig l.47] fraction of a spike sent to the next gates (rest -> shunt) | [schematic] L2/R1 division of loop 4 | default 0.1
parser.add_argument("--split", help="current splitting fraction", type=float, default=0.1 )
# [orig l.48] use the time-of-flight generator to set the gap between the two sensor clicks
parser.add_argument("--generator", help="use tof generator" , action='store_true' )
# [orig l.49] gap between the two sensor clicks (ns) | [schematic] pulser2 (U4) fires this long after pulser1 (U1) | default 5
parser.add_argument("--dt", help="delta t in ns, ignored if generator is used", type=float, default=5 )

# [orig l.51] per-node input bias, repeatable: --ib1 NODE VALUE, e.g. --ib1 2 13 sets t1 to 13 uA
#   NODE = row number of the delay table: 0 = i1, 1 = i2, 2 = t1 ... 7 = t6. [schematic] bias3 of one box
parser.add_argument('--ib1', nargs=2, action='append', type=float, help='set node [0] input bias to value [1]')
# [orig l.52] same for the output bias (bias4)
parser.add_argument('--ib2', nargs=2, action='append', type=float, help='set node [0] output bias to value [1]')
# [new] which node to draw the gate-current trace for (default 4 = t3)
parser.add_argument('--trace_node', type=int, default=4, help='matrix index of node for the gate-current trace plot (0=i1, 1=i2, 2=t1 ... 7=t6); default t3')
# [orig l.54] read the flags from the command line
args = parser.parse_args()

# [orig l.56] debug print of the --ib1 list
print(args.ib1)


# ---------------- STEP 4: convert flags to SI units and set the fixed circuit values ----------------
# [orig l.58] U3 threshold, uA -> A
Inw1_sw = args.I1sw*1.0e-6
# [orig l.59] U2 threshold, uA -> A
Inw2_sw = args.I2sw*1.0e-6

# [orig l.61] input bias, uA -> A   [schematic] I_b1
Ib1 = args.Ib1init*1.0e-6
# [orig l.62] output bias, uA -> A  [schematic] I_b2
Ib2 = args.Ib2init*1.0e-6

# [orig l.64] fixed circuit values (not flags)
# [orig l.65] inductance of the shunt around U3 | [schematic] L1, 5 nH, loop 2
Ls1 = 5 * 1.0e-9
# [orig l.66] resistance of that shunt (ohm) | [schematic] R3, loop 2 (3 ohm in LTspice, 2 here)
Rs1 = 2
# [orig l.67] storage-loop inductance | [schematic] Lb1 ~148 nH, loop 3
LT  = 150 * 1.0e-9
# [orig l.68] storage-loop resistance when U2 is not switched (ohm) | [schematic] R2, loop 3 (4 ohm in LTspice, 3 here)
RT0 = 3

# [orig l.70] inductance of an nTron channel | [schematic] L_nw inside U3 / U2
Lnw   = 10 * 1.0e-9
# [orig l.71] resistance of an nTron channel when switched (ohm) | [schematic] R_nw(t) of U3; for U2 it is added onto R2
Rnorm = 1000

# [orig l.73] unused variable (the original's own comment says so)
recover_time1 = 3.0e-9
# [orig l.74-75] rest time (3 ns) after a click before that nTron can click again. The original's comment says
#   'also not used??' but it IS used: in RHS at orig l.187 and l.229
ref_period = 0.003 *1.0e-6



# ---------------- STEP 5: the wiring of the network ----------------
# [orig l.79] rows = senders, columns = receivers; rows/columns 0-1 are the sensors i1, i2, then t1...t6
#  first two columns are inputs   
##                       to: i1  i2  t1  t2  t3  t4  t5  t6    from:
# [orig l.81-89] THE NETWORK IN ONE TABLE: entry = delay in ns of the link sender -> receiver (0 = no link).
#   CHANGED from the original (which was feedforward only: t1->t3, t2->t3, t3->t4; t4-t6 sent nothing)
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

# [orig l.94] prints (8, 8)
print(np.shape(delay))
# [orig l.95] prints 8
print(len(delay))

# [orig l.97] number of nodes = number of rows (8)
N_transponders = len(delay)


# ---------------- STEP 6: starting bookkeeping (nothing has fired yet) ----------------
# [orig l.99] state codes: 0 = ready / superconducting, 1 = clicked
# [orig l.100] every U3 starts ready
input_states = np.zeros(N_transponders)
# [orig l.101] every U2 starts ready
output_states = np.zeros(N_transponders)

# [orig l.103] history of input clicks, one empty list per node
input_click_times = [[] for _ in range(N_transponders)]
# [orig l.104] history of spikes, one empty list per node
output_click_times = [[] for _ in range(N_transponders)]

# [orig l.106] sensor gap, ns -> s
dt = 1.0e-9*args.dt
# [orig l.107-110] optionally take the gap from the time-of-flight generator instead
if args.generator:
  print("using generator")
  ev = tof_gen.generate_tof_event()
  dt = ev["tof"]

# [orig l.112] when each sensor clicks: i1 at 1 ns, i2 at 1 ns + dt   [schematic] pulser1 (U1), pulser2 (U4)
sensor_pulses = [[1.0e-9],[1.0e-9 + dt]]

# [orig l.114] every node gets the same input bias (bias3) ...
transponder_bias1  = Ib1*np.ones(N_transponders)
# [orig l.115] ... and the same output bias (bias4)
transponder_bias2  = Ib2*np.ones(N_transponders)

# [orig l.117-120] then apply the per-node --ib1 exceptions (value uA -> A)
if args.ib1:
  for setting in args.ib1:
    transponder_bias1[int(setting[0])] = setting[1]*1.0e-6
# [orig l.121-123] same for --ib2
if args.ib2:
  for setting in args.ib2:
    transponder_bias2[int(setting[0])] = setting[1] *1.0e-6

# [orig l.125] the split rule
# [orig l.126] fraction of a spike's current sent on to the next gates | [schematic] L2/R1 division; used ONLY in RHS step 3 (orig l.210)
transponder_current_divider = args.split

# [orig l.128] values that RHS updates at every step
# [orig l.129] U3's resistance R_nw1(t): 0 = superconducting
transponder_Rnw_input = np.zeros(N_transponders)
# [orig l.130] loop 3's resistance R_T(t): R2, plus U2 while it is switched
transponder_RT_loop   = RT0*np.ones(N_transponders)
# [orig l.131] record of the gate current (for plotting; not used in the physics)
transponder_input_gate_current = np.zeros(N_transponders)
# [orig l.132] record of i_nw + gate current (for plotting)
transponder_source_current = np.zeros(N_transponders)

# [orig l.134] starting U3 channel currents = the bias currents (each U3 carries its whole bias at rest)
init_input_current  = transponder_bias1
# [orig l.135] storage loops start empty
init_loop_currents  = np.zeros(N_transponders)


# [orig l.138] the vector handed to the solver: 2 slots per node = 16 (even = i_nw, odd = i_loop)
transponder_initial_state =  np.empty((2*N_transponders), dtype=transponder_RT_loop.dtype)
# [orig l.139] even slots: i_nw = bias
transponder_initial_state[0::2] = init_input_current
# [orig l.140] odd slots: i_loop = 0
transponder_initial_state[1::2] = init_loop_currents


# ---------------- STEP 7: flip the wiring table to 'who sends to me?' ----------------
# [orig l.143]
# [orig l.144] one empty list per receiver
input_connections = [[] for _ in range(len(input_states))] 
# [orig l.145] transpose: row = receiver, column = sender
input_delays = delay.T
# [orig l.146] for each receiver i ...
for i,arow in enumerate(connectivity.T):
  # [orig l.147] ... and each possible sender j
  for j,acol in enumerate(arow):
    # [orig l.148] if a link exists
    if acol == 1:
      # [orig l.149] remember that j sends to i
      input_connections[i].append(j)


# [orig l.154] (re-initialises output_states; harmless duplicate of l.101)
output_states = np.zeros(N_transponders)
# [orig l.156] (re-initialises the spike histories; harmless duplicate of l.104)
output_click_times = [[] for _ in range(len(input_states))]



# ---------------- STEP 8: RHS, the engine ----------------
# [orig l.159] The solver calls this at every tiny time step with the time (t) and all 16 currents (x),
# and gets back how fast each current is changing. Everything below is evaluated at that instant.
def RHS(t, x):
  # [orig l.162] even entries of x: U3 channel current of each node
  nw_currents   = x[::2]  
  # [orig l.163] odd entries: storage-loop current of each node
  loop_currents = x[1::2]

  # [orig l.165] 'close enough' for matching clock times: 10 ps
  eps = 1.0e-11

  # ===== RHS step 1 - the sensors fire (orig l.167-175): the outside world pokes the network =====
  # [orig l.167] for each sensor and ...
  for n, clicks in enumerate(sensor_pulses):
    # [orig l.168] ... each click time it still has scheduled
    for i_click, sensor_click_t in enumerate(clicks):  
      # [orig l.169] is it that time now (within 10 ps)?
      if( abs(t -sensor_click_t) <= eps ) :
        # [orig l.170] debug print
        print(f"n = {n} i_click = {i_click} time {t} vs {sensor_click_t}")
        # [orig l.171] mark the sensor 'clicked'
        output_states[n] = 1
        # [orig l.174] RECORD THE CLICK TIME: every node listening to this sensor will find it in RHS step 3
        output_click_times[n].append(t)
        # [orig l.175] remove it from the schedule so it fires once
        sensor_pulses[n].pop(i_click)

  # ===== RHS step 2 - output nTron U2 (orig l.177-197): did each transponder fire? =====
  # [orig l.178] for each node
  for i_out, output_state in enumerate(output_states):
    # [orig l.179] U2 is currently switched (fired recently):
    if output_states[i_out] == 1:
      # [orig l.183] when did it last fire?
      tlast                       = output_click_times[i_out][len(output_click_times[i_out])-1]
      # [orig l.185] while switched: loop resistance = R2 + a 1000 ohm pulse (2 ns). This dumps the stored loop current
      transponder_RT_loop[i_out]  = RT0 + Rnorm*LinearPulse(t,t0=tlast,width=2.0e-9)
      # [orig l.187] after ref_period (3 ns) ...
      if t-tlast > ref_period:
        # [orig l.188] ... U2 may fire again
        output_states[i_out] = 0
        # [orig l.189] debug print
        print("resetting output ntron")
    # [orig l.190] U2 is ready:
    if output_states[i_out] == 0:
      # [orig l.191] loop resistance is just R2
      transponder_RT_loop[i_out] = RT0
      # [orig l.193] FIRING TEST: storage-loop current (U2's gate) + bias4 > U2's threshold?
      if loop_currents[i_out] + transponder_bias2[i_out] > Inw2_sw: 
        # [orig l.195] debug print
        print("output {}: click_t = {}".format(i_out, t))
        # [orig l.196] yes: U2 switches
        output_states[i_out] = 1
        # [orig l.197] RECORD THE SPIKE TIME. This is the moment current leaves the transponder
        output_click_times[i_out].append(t)

  # ===== RHS step 3 - input nTron U3 (orig l.199-236): did enough arrive from upstream? =====
  # [orig l.201] for each node
  for i_in, input_state in enumerate(input_states):
    # [orig l.202] U3 is ready:
    if input_states[i_in] == 0:
      # [orig l.203] superconducting: no resistance
      transponder_Rnw_input[i_in] = 0
      # [orig l.205] start summing the gate current from zero
      ntron_gate_current = 0
      # [orig l.206] (record, for plotting)
      transponder_input_gate_current[i_in] = 0
      # [orig l.207] look at every node that sends to this one
      for j_in in input_connections[i_in]:
        # [orig l.208] how many nodes that sender feeds (its fan-out)
        n_split = np.sum(connectivity[j_in])
        # [orig l.209] the sender's output bias (bias4)
        b2  = transponder_bias2[j_in]
        # [orig l.210] PULSE SIZE from this sender = Ib2 x split / fan-out (the only use of `split`)
        amp = b2*transponder_current_divider/n_split
        # [orig l.212] for every spike that sender has ever fired ...
        for click_t in output_click_times[j_in]:
          # [orig l.213] the link delay (the delay box in the schematic) ...
          click_delay         = input_delays[i_in][j_in]
          # [orig l.214] ... so the spike arrives here at spike time + delay
          click_arrival       = click_t  + click_delay
          # [orig l.215] add that pulse's current right now to the gate current (loop 1). Pulses from several senders add up
          ntron_gate_current += amp*OutputPulse(t, t0=click_arrival)
      # [orig l.218] (record i_nw + gate, for plotting)
      transponder_source_current[i_in]     = nw_currents[i_in] + ntron_gate_current
      # [orig l.219] (record gate current, for plotting)
      transponder_input_gate_current[i_in] = ntron_gate_current
      # [orig l.220] CLICK TEST: U3 channel current + gate current > U3's threshold?
      if nw_currents[i_in] + ntron_gate_current > Inw1_sw: 
        # [orig l.222] yes: U3 clicks
        input_states[i_in] = 1
        # [orig l.223] record the click time
        input_click_times[i_in].append(t)
    # [orig l.226] U3 is clicked (resistive):
    if input_states[i_in] == 1:
      # [orig l.228] when did it click?
      latest_input_click = input_click_times[i_in][len(input_click_times[i_in])-1]
      # [orig l.229] after 3 ns ...
      if t-latest_input_click > ref_period:
        # [orig l.231] debug print
        print("input reset")
        # [orig l.232] ... U3 is ready again
        input_states[i_in] = 0
      # [orig l.234] otherwise still resistive:
      else: 
        # [orig l.236] U3's resistance: 1000 ohm trapezoid (about 2.5 ns). This pushes current out of U3 into the other branches
        transponder_Rnw_input[i_in] = Rnorm*LinearPulse(t,t0=latest_input_click, width=1.0e-9)
  # ===== RHS step 4 - circuit physics (orig l.237-250): where does the current go? =====
  # [orig l.237] list of rates of change we will return
  y = []
  # [orig l.238] for each node
  for i_node,loop_cur in enumerate(loop_currents):
    # [orig l.239] its input bias (bias3) flowing in
    i1  = transponder_bias1[i_node]
    # [orig l.240] current in U3 right now
    inw = nw_currents[i_node]
    # [orig l.241] current in the storage loop right now
    iloop = loop_cur
    # [orig l.242] U3's resistance right now (from RHS step 3)
    Rnw =  transponder_Rnw_input[i_node]
    # [orig l.243] loop 3's resistance right now (from RHS step 2)
    RT  = transponder_RT_loop[i_node]
    # [orig l.246] d(i_nw)/dt: how fast the U3 current changes (Kirchhoff + inductors)
    y1 = (-inw*Ls1*Rnw - i1*LT*Rs1 + iloop*LT*Rs1 + inw*LT*(Rnw + Rs1) + iloop*Ls1*RT)/(Lnw*(Ls1 - LT) - Ls1*LT)
    # [orig l.247] d(i_loop)/dt: how fast the storage-loop current changes
    y2 = (-inw*Ls1*Rnw - i1*Lnw*Rs1 + iloop*Lnw*Rs1 + inw*Lnw*Rs1 + iloop*Lnw*RT + iloop*Ls1*RT)/(Lnw*(Ls1 - LT) - Ls1*LT)
    # [orig l.248-249] store both
    y.append(y1)
    y.append(y2)
  # [orig l.250] hand all 16 rates back to the solver, which then takes the next step and calls RHS again
  return y




# ---------------- STEP 9: main(), run the simulation and plot ----------------
# [orig l.254]
def main() -> int:
  """ Prototype network simulation"""
  # [orig l.256] print the wiring table
  print(delay)
  


  # [orig l.271] the 1000 times at which results are saved (0-100 ns)
  t_pts = np.linspace(0, 100*1.0e-9, num=1000)

  # [orig l.273] print the split
  print(f"shunt/output current splilt: {transponder_current_divider}")

  # [orig l.275]
  print("Starting simulation")
  # [orig l.276] RUN IT: start at 0, end at 100 ns, begin with the initial currents; step <= 1 ps so no switching event is skipped (slow, ~1 min)
  solution = solve_ivp(RHS, (0, 100 * 1.0e-9), transponder_initial_state, t_eval= t_pts, max_step = 1.0e-12)

  # [orig l.278] the original 6-panel plot: for t1, t2, t3 (rows 4-9 of the solution) the U3 current (labelled 'gate') and the loop current (red)
  fig, axs = plt.subplots(6)
  # [orig l.280-285] solution.y[2n] = i_nw of node n, solution.y[2n+1] = i_loop (uA)
  axs[0].plot(solution.t/1.0e-9, solution.y[4]/1.0e-6, label='gate 2')
  axs[1].plot(solution.t/1.0e-9, solution.y[5]/1.0e-6, 'r',label='loop 2')
  axs[2].plot(solution.t/1.0e-9, solution.y[6]/1.0e-6, label='gate 3')
  axs[3].plot(solution.t/1.0e-9, solution.y[7]/1.0e-6, 'r',label='loop 3')
  axs[4].plot(solution.t/1.0e-9, solution.y[8]/1.0e-6, label='gate 4')
  axs[5].plot(solution.t/1.0e-9, solution.y[9]/1.0e-6, 'r',label='loop 4')

  # [orig l.291] write dt and the biases on the figure
  axs[0].annotate(f"dt={dt//1.0e-9}, \nb1={transponder_bias1/1.0e-6}, \nb2={transponder_bias2/1.0e-6}", xycoords="subfigure points",xy=(1,1))

  # [orig l.293-298] legends
  axs[0].legend()
  axs[1].legend()
  axs[2].legend()
  axs[3].legend()
  axs[4].legend()
  axs[5].legend()

  # [orig l.300-301] axis labels
  for ax in axs.flat:
    ax.set(xlabel='time [ns]',ylabel='current [uA]')

  # [orig l.303-305] hide repeated labels
  for ax in axs.flat:
    ax.label_outer()
  # [orig l.307-308] save the figure
  plt.savefig(f"../results/{args.output}.png", bbox_inches='tight',dpi=200)
  plt.savefig(f"../results/{args.output}.pdf", format="pdf", bbox_inches='tight',dpi=200)

  # [orig l.311] 0 = solver finished OK
  print(f"solution status {solution.status}")


  # ---------------- STEP 10 [new]: rebuild the gate current, then the three plots you asked for ----------------
  # [new] time axis in seconds ...
  t   = solution.t
  # [new] ... and in ns
  tns = t/1.0e-9
  # [new] node labels
  names = ['i1', 'i2'] + [f't{k}' for k in range(1, N_transponders-1)]
  # [new] gate_parts[i][j] = current node i receives from j; gate_current[i] = the sum (RHS computed it but threw it away)
  gate_parts = [dict() for _ in range(N_transponders)]
  # [new] one row per node, one column per saved time
  gate_current = np.zeros((N_transponders, len(t)))
  # [new] for each receiver ...
  for i_in in range(N_transponders):
    # [new] ... and each sender feeding it
    for j_in in input_connections[i_in]:
      # [new] pulse size = sender's Ib2 x split / fan-out (same as orig l.208-210)
      amp  = transponder_bias2[j_in]*transponder_current_divider/np.sum(connectivity[j_in])
      # [new] this sender's contribution over time
      part = np.zeros(len(t))
      # [new] for each spike the sender fired ...
      for click_t in output_click_times[j_in]:
        # [new] ... it arrives after the link delay (orig l.213-214)
        arrival = click_t + input_delays[i_in][j_in]
        # [new] add the pulse at every saved time, only after the spike happened
        part += np.array([amp*OutputPulse(tk, t0=arrival) if tk >= click_t else 0.0 for tk in t])
      # [new] keep this sender's piece
      gate_parts[i_in][j_in] = part
      # [new] and add to the total
      gate_current[i_in] += part

  # [new] print every node's spike times (ns)
  print("output (spike) click times [ns]:")
  for n in range(N_transponders):
    print(f"  {names[n]}: {np.round(np.array(output_click_times[n])/1.0e-9, 2)}")

  # [new] SPIKE RASTER: one row per node, black tick = spike, red triangle = input click
  fig, ax = plt.subplots(figsize=(8, 3.5))
  # [new] for each node
  for n in range(N_transponders):
    # [new] black tick at each spike time
    ax.vlines(np.array(output_click_times[n])/1.0e-9, n-0.4, n+0.4, color='k', lw=2,
              label='output click (spike)' if n == 0 else None)
    # [new] red triangle at each input click
    if input_click_times[n]:
      ax.plot(np.array(input_click_times[n])/1.0e-9, [n-0.45]*len(input_click_times[n]), 'v', color='tab:red', ms=4,
              label='input (gate) click' if not any(input_click_times[:n]) else None)
  # [new] dashed separators: sensors | t1-t3 | t4-t6
  for y in (1.5, 4.5):
    ax.axhline(y, color='0.8', ls='--', lw=0.8)
  # [new] row labels
  ax.set_yticks(range(N_transponders), names)
  # [new] i1 at the top
  ax.set_ylim(N_transponders-0.5, -0.5)
  # [new] axes and title
  ax.set(xlim=(0, tns[-1]), xlabel='time [ns]', title='spike raster')
  # [new] legend
  ax.legend(loc='upper right', fontsize=8, frameon=False)
  # [new] save raster
  plt.savefig(f"../results/{args.output}_raster.png", bbox_inches='tight', dpi=200)
  plt.savefig(f"../results/{args.output}_raster.pdf", bbox_inches='tight')

  # [new] GATE-CURRENT TRACE for the node chosen with --trace_node: do the pulses sum above threshold?
  n = args.trace_node
  # [new] its U3 channel current over time
  inw = solution.y[2*n]
  # [new] two stacked panels
  fig, axs2 = plt.subplots(2, sharex=True, figsize=(8, 5))
  # [new] TOP: each sender's pulse (dashed) ...
  for j_in, part in gate_parts[n].items():
    axs2[0].plot(tns, part/1.0e-6, '--', label=f'from {names[j_in]}')
  # [new] ... and their sum (black)
  axs2[0].plot(tns, gate_current[n]/1.0e-6, 'k', lw=1.5, label='summed gate current')
  # [new] red dotted: gate current needed to click (threshold - bias)
  axs2[0].axhline((Inw1_sw - transponder_bias1[n])/1.0e-6, color='r', ls=':', label='needed to switch (I1sw - Ib1)')
  # [new] labels
  axs2[0].set(ylabel='gate current [uA]', title=f'{names[n]}: gate current')
  # [new] legend
  axs2[0].legend(fontsize=8, frameon=False)
  # [new] BOTTOM: grey = U3 channel current ...
  axs2[1].plot(tns, inw/1.0e-6, color='0.5', label='channel current i_nw')
  # [new] ... black = channel + gate current: what the click test compares (orig l.220)
  axs2[1].plot(tns, (inw + gate_current[n])/1.0e-6, 'k', label='i_nw + gate')
  # [new] red dotted = U3 threshold
  axs2[1].axhline(Inw1_sw/1.0e-6, color='r', ls=':', label='I1sw (threshold)')
  # [new] faint red lines at this node's input clicks
  for k, tc in enumerate(input_click_times[n]):
    axs2[1].axvline(tc/1.0e-9, color='r', alpha=0.4, label='input click' if k == 0 else None)
  # [new] labels
  axs2[1].set(xlabel='time [ns]', ylabel='current [uA]')
  # [new] legend
  axs2[1].legend(fontsize=8, frameon=False)
  # [new] save gate trace
  plt.savefig(f"../results/{args.output}_gate_trace_{names[n]}.png", bbox_inches='tight', dpi=200)
  plt.savefig(f"../results/{args.output}_gate_trace_{names[n]}.pdf", bbox_inches='tight')

  # [new] GRADIENT RASTER: shade = gate current (white = none, black = the largest in the run); white again right after a click
  # [new] largest gate current anywhere (avoid dividing by 0)
  gmax  = gate_current.max() if gate_current.max() > 0 else 1.0
  # [new] scale every value to 0..1
  shade = gate_current/gmax
  # [new] for each node ...
  for n in range(N_transponders):
    # [new] ... and each input click ...
    for tc in input_click_times[n]:
      # [new] ... paint white for ref_period afterwards (node recovering)
      shade[n][(t > tc) & (t <= tc + ref_period)] = 0
  # [new] figure
  fig, ax = plt.subplots(figsize=(8, 3.5))
  # [new] draw the shading as an image (rows = nodes, columns = time)
  im = ax.imshow(shade, aspect='auto', cmap='Greys', vmin=0, vmax=1, interpolation='nearest',
                 extent=(tns[0], tns[-1], N_transponders-0.5, -0.5))
  # [new] overlay the clicks as red lines
  for n in range(N_transponders):
    # [new] solid red = spike
    ax.vlines(np.array(output_click_times[n])/1.0e-9, n-0.5, n+0.5, color='red', lw=1.8,
              label='output click (spike)' if n == 0 else None)
    # [new] dotted red = input click
    ax.vlines(np.array(input_click_times[n])/1.0e-9, n-0.5, n+0.5, color='red', lw=0.8, ls=':',
              label='input (gate) click' if n == 2 else None)
  # [new] legend below the plot
  ax.legend(loc='upper center', fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.15))
  # [new] separators
  for y in (1.5, 4.5):
    ax.axhline(y, color='0.8', ls='--', lw=0.8)
  # [new] row labels
  ax.set_yticks(range(N_transponders), names)
  # [new] axis label and title
  ax.set(xlabel='time [ns]', title='gradient raster: gate current (white = 0, black = max)')
  # [new] colour bar: shade -> fraction of the largest gate current
  fig.colorbar(im, ax=ax, label=f'gate current / {gmax/1.0e-6:.2f} uA', pad=0.01)
  # [new] save gradient raster
  plt.savefig(f"../results/{args.output}_gradient_raster.png", bbox_inches='tight', dpi=200)
  plt.savefig(f"../results/{args.output}_gradient_raster.pdf", bbox_inches='tight')

  # [orig l.314] success
  return 0

# [orig l.316-317] run main() when the file is executed
if __name__ == '__main__':
  sys.exit(main())
