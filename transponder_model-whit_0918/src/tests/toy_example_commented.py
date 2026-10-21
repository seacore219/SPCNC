# ======================================================================================================
# COMMENTED COPY of toy_example.py  (same code, comments only; the code is checked to be identical)
#
# Comment key. Above each line of code:
#   [orig l.N]      what the line says, translated word for word into English. N = line number in
#                   toy_example.py on `main` (Whit's original, 317 lines). [new] = code not in the original
#                   (your requests: recurrent wiring, raster, gate trace, gradient raster)
#   significance:   why the line matters / which part of the schematic it is
#                   (single_transponder_simple.pdf: U3 = input nTron (Q1), U2 = output nTron (Q2),
#                   bias3 = I_b1, bias4 = I_b2, loop 1 = input/gate loop, loop 2 = Q1 shunt (L1, R3),
#                   loop 3 = storage loop (Lb1, R2), loop 4 = output loop (L2, R1))
# Comments inside the delay table are yours and are kept.
#
# How the program runs: (1) read flags, (2) set up the network tables and starting currents,
# (3) RHS() says how fast every current changes at an instant, (4) main() hands RHS to a solver that
# steps through 100 ns, (5) plots.
# ======================================================================================================

# ---------------- STEP 1: libraries ----------------
# [orig l.1] load the numpy library and call it np
# significance: arrays: the network tables and the current vectors
import numpy as np
# [orig l.2] from the scipy.integrate module, load the function solve_ivp
# significance: the solver that steps the circuit equations forward in time
from scipy.integrate import solve_ivp
# [orig l.3] load the matplotlib.pyplot plotting library and call it plt
import matplotlib.pyplot as plt
# [orig l.4] load the sys library
# significance: only used on the last line, to return the exit status
import sys
# [orig l.5] load the math library
# significance: gives exp() for OutputPulse
import math
# [orig l.6] load the argparse library
# significance: reads the command-line flags
import argparse
# [orig l.7] load the module spcnc.tof_input_generator and call it tof_gen
# significance: only used when you run with --generator
import spcnc.tof_input_generator as tof_gen

# [orig l.9] create a flag reader called parser, with the program name 'spcnc_toy_example' and the description
#            'prototype spiking network simulation'
parser = argparse.ArgumentParser(
                    prog='spcnc_toy_example',
                    description='prototype spiking network simulation',
                    epilog='')


# ---------------- STEP 2: the two pulse shapes ----------------
# [orig l.14] define a function LinearPulse. Its inputs are t, t0 (default 5 ns), width (default 0.5 ns), rise
#             (default 1 ns) and fall (default 0.5 ns)
# significance: a trapezoid between 0 and 1. Multiplied by Rnorm it is an nTron's resistance while the nTron is
#               switched. t = now, t0 = when it starts
def LinearPulse(t, t0 = 5.0e-9, width=0.5e-9, rise=1.0e-9, fall=0.5e-9):
  # [orig l.15] IF (t minus t0) is less than 0, OR (t minus t0) is greater than (width + fall + rise)
  # significance: before the pulse has started or after it has finished
  if (t-t0 < 0) or (t-t0 > width+fall+rise): 
    # [orig l.16] RETURN 0
    # significance: the nTron is superconducting (zero resistance)
    return 0
  # [orig l.17] IF (t minus t0) is greater than 0, AND (t minus t0) is less than rise
  # significance: we are on the rising edge
  if ((t-t0) > 0) and ((t - t0) < rise):
    # [orig l.19] set delta_t equal to t minus t0
    # significance: how long since the pulse started
    delta_t = t-t0
    # [orig l.21] RETURN delta_t divided by rise
    # significance: a straight line going from 0 up to 1
    return delta_t/rise 
  # [orig l.22] IF (t minus t0) is greater than (width + rise), AND (t minus t0) is less than or equal to (width +
  #             fall + rise)
  # significance: we are on the falling edge
  if (t-t0 > width+rise) and (t-t0 <= width+fall+rise):
    # [orig l.24] set delta_t equal to (t0 + width + fall + rise) minus t
    # significance: how long until the pulse ends
    delta_t = (t0+width+fall+rise)-t
    # [orig l.26] RETURN delta_t divided by fall
    # significance: a straight line going from 1 down to 0
    return (delta_t/fall)
  # [orig l.27] OTHERWISE
  # significance: we are on the flat top
  else: 
   # [orig l.28] RETURN 1
   # significance: full resistance
   return 1

# [orig l.30] define a function OutputPulse. Its inputs are t, t0 (default 1 ns), tau (default 1 ns) and mu0 (default
#             0.2 ns)
# significance: the shape of the current one upstream spike delivers into a node's gate (loop 1). t0 = where it is
#               about half way up, tau = decay time, mu0 = how sharp the rise is
def OutputPulse(t, t0=1e-9, tau=1.0e-9, mu0=0.2e-9):
  # [orig l.33] RETURN (1 minus 1 divided by (e to the power ((t minus t0) divided by mu0) plus 1)) TIMES e to the
  #             power (minus (t minus t0) divided by tau)
  # significance: a smooth step up around t0 multiplied by an exponential decay. It peaks at about 0.61. Because the
  #               step is smooth it is already about 20% of its amplitude 0.35 ns BEFORE t0
  return (1.0 - 1.0/(math.exp((t - t0)/mu0) + 1.0))*math.exp(- ((t - t0)/tau))

"""
Units: can we default to us? 

There are two input pulses of fixed shape.
This defines the input currents 
"""

# ---------------- STEP 3: the command-line flags (the knobs) ----------------
# [orig l.42] add a flag --output that takes text, default 'toy_example'
# significance: base name of the plot files written to ../results
parser.add_argument("--output", help="output plots file name", default="toy_example" )
# [orig l.43] add a flag --Ib1init that takes a number, default 16
# significance: input bias of every node (uA). Schematic: I_b1, bias3 into U3's drain (LTspice uses ~150 uA)
parser.add_argument("--Ib1init", help="I_b1 initial value in uA", type=float, default=16 )
# [orig l.44] add a flag --Ib2init that takes a number, default 16
# significance: output bias of every node (uA). Schematic: I_b2, bias4 into U2's drain (LTspice uses ~160 uA)
parser.add_argument("--Ib2init", help="I_b1 initial value in uA", type=float, default=16 )
# [orig l.45] add a flag --I1sw that takes a number, default 20
# significance: switching current of U3 = the input threshold (uA)
parser.add_argument("--I1sw", help="input ntron switching current in uA", type=float, default=20 )
# [orig l.46] add a flag --I2sw that takes a number, default 20
# significance: switching current of U2 = the output threshold (uA)
parser.add_argument("--I2sw", help="output ntron switching current in uA", type=float, default=20 )
# [orig l.47] add a flag --split that takes a number, default 0.1
# significance: fraction of a spike's current sent on to the next gates; the rest goes to the shunt. Schematic: the L2
#               / R1 division of loop 4
parser.add_argument("--split", help="current splitting fraction", type=float, default=0.1 )
# [orig l.48] add a flag --generator that is just on/off, default off
# significance: use the time-of-flight generator to pick the gap between the two sensor clicks
parser.add_argument("--generator", help="use tof generator" , action='store_true' )
# [orig l.49] add a flag --dt that takes a number, default 5
# significance: gap in ns between the two sensor clicks. Schematic: pulser2 (U4) fires this long after pulser1 (U1)
parser.add_argument("--dt", help="delta t in ns, ignored if generator is used", type=float, default=5 )

# [orig l.51] add a flag --ib1 that takes exactly 2 numbers and can be repeated
# significance: --ib1 NODE VALUE sets one node's input bias. NODE = row of the delay table (0 = i1, 1 = i2, 2 = t1 ...
#               7 = t6); VALUE in uA. Example: --ib1 2 13 sets t1 to 13 uA
parser.add_argument('--ib1', nargs=2, action='append', type=float, help='set node [0] input bias to value [1]')
# [orig l.52] add a flag --ib2 that takes exactly 2 numbers and can be repeated
# significance: same as --ib1 but for the output bias (bias4)
parser.add_argument('--ib2', nargs=2, action='append', type=float, help='set node [0] output bias to value [1]')
# [new] add a flag --trace_node that takes a whole number, default 4
# significance: which node the gate-current trace is drawn for (4 = t3)
parser.add_argument('--trace_node', type=int, default=4, help='matrix index of node for the gate-current trace plot (0=i1, 1=i2, 2=t1 ... 7=t6); default t3')
# [orig l.54] read the command line and store every flag's value in args
args = parser.parse_args()

# [orig l.56] print the list of --ib1 settings
# significance: debug output
print(args.ib1)


# ---------------- STEP 4: convert flags to SI units and set the fixed circuit values ----------------
# [orig l.58] set Inw1_sw equal to args.I1sw times 1e-6
# significance: microamps to amps. This is U3's threshold
Inw1_sw = args.I1sw*1.0e-6
# [orig l.59] set Inw2_sw equal to args.I2sw times 1e-6
# significance: microamps to amps. This is U2's threshold
Inw2_sw = args.I2sw*1.0e-6

# [orig l.61] set Ib1 equal to args.Ib1init times 1e-6
# significance: microamps to amps. Schematic: I_b1 (bias3)
Ib1 = args.Ib1init*1.0e-6
# [orig l.62] set Ib2 equal to args.Ib2init times 1e-6
# significance: microamps to amps. Schematic: I_b2 (bias4)
Ib2 = args.Ib2init*1.0e-6

# [orig l.65] set Ls1 equal to 5 times 1e-9
# significance: 5 nH, the inductance of the shunt around U3. Schematic: L1 (loop 2)
Ls1 = 5 * 1.0e-9
# [orig l.66] set Rs1 equal to 2
# significance: 2 ohm, the resistance of that shunt. Schematic: R3 (loop 2); it is 3 ohm in LTspice
Rs1 = 2
# [orig l.67] set LT equal to 150 times 1e-9
# significance: 150 nH, the storage-loop inductance. Schematic: Lb1 (~148 nH, loop 3)
LT  = 150 * 1.0e-9
# [orig l.68] set RT0 equal to 3
# significance: 3 ohm, the storage-loop resistance while U2 is not switched. Schematic: R2 (loop 3); it is 4 ohm in
#               LTspice
RT0 = 3

# [orig l.70] set Lnw equal to 10 times 1e-9
# significance: 10 nH, the inductance of an nTron channel. Schematic: L_nw inside U3 and U2
Lnw   = 10 * 1.0e-9
# [orig l.71] set Rnorm equal to 1000
# significance: 1000 ohm, an nTron channel's resistance when switched. Schematic: R_nw(t) of U3; for U2 it is added
#               onto R2
Rnorm = 1000

# [orig l.73] set recover_time1 equal to 3 times 1e-9
# significance: never used anywhere (the original's own comment says so)
recover_time1 = 3.0e-9
# [orig l.75] set ref_period equal to 0.003 times 1e-6
# significance: 3 ns: rest time after a click before that nTron can click again. (The original comment says 'not
#               used??' but it IS used, at orig l.187 and l.229)
ref_period = 0.003 *1.0e-6



# ---------------- STEP 5: the wiring of the network ----------------
# [orig l.79] (next, the wiring table. Rows = senders, columns = receivers)
# significance: rows and columns 0-1 are the sensors i1, i2, then t1...t6
#  first two columns are inputs   
##                       to: i1  i2  t1  t2  t3  t4  t5  t6    from:
# [orig l.81] set delay equal to 1e-9 times a table of 8 rows by 8 columns of numbers (the table is written out below)
# significance: THE NETWORK IN ONE TABLE. Entry [row][column] = delay in ns of the link from the row node to the
#               column node, 0 = no link. CHANGED from the original, which was feedforward only (t1->t3, t2->t3,
#               t3->t4; t4-t6 sent nothing)
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
# [orig l.92] set connectivity equal to the smaller of 1 and the rounded-up value of delay, entry by entry
# significance: 1 where a link exists, 0 where there is none
connectivity = np.minimum(1,np.ceil(delay)) # converts binarizes delay function to either 0 or 1 (connnection or no connection)

# [orig l.94] print the shape of delay
# significance: prints (8, 8)
print(np.shape(delay))
# [orig l.95] print the number of rows of delay
# significance: prints 8
print(len(delay))

# [orig l.97] set N_transponders equal to the number of rows of delay
# significance: 8 nodes (2 sensors + 6 transponders)
N_transponders = len(delay)


# ---------------- STEP 6: starting bookkeeping (nothing has fired yet) ----------------
# [orig l.100] set input_states equal to a list of N_transponders zeros
# significance: state of every U3: 0 = ready (superconducting), 1 = clicked. All start ready
input_states = np.zeros(N_transponders)
# [orig l.101] set output_states equal to a list of N_transponders zeros
# significance: state of every U2 (same codes). All start ready
output_states = np.zeros(N_transponders)

# [orig l.103] set input_click_times equal to an empty list for each node
# significance: the history of when each node's U3 clicked
input_click_times = [[] for _ in range(N_transponders)]
# [orig l.104] set output_click_times equal to an empty list for each node
# significance: the history of when each node fired (its spike times)
output_click_times = [[] for _ in range(N_transponders)]

# [orig l.106] set dt equal to 1e-9 times args.dt
# significance: ns to seconds: the gap between the two sensor clicks
dt = 1.0e-9*args.dt
# [orig l.107] IF the --generator flag was given
if args.generator:
  # [orig l.108] print 'using generator'
  print("using generator")
  # [orig l.109] set ev equal to a new event from the time-of-flight generator
  ev = tof_gen.generate_tof_event()
  # [orig l.110] set dt equal to the event's 'tof' value
  # significance: the generator overrides the gap between the sensor clicks
  dt = ev["tof"]

# [orig l.112] set sensor_pulses equal to two lists: [1 ns] and [1 ns + dt]
# significance: when each sensor clicks: i1 at 1 ns, i2 at 1 ns + dt. Schematic: pulser1 (U1) and pulser2 (U4)
sensor_pulses = [[1.0e-9],[1.0e-9 + dt]]

# [orig l.114] set transponder_bias1 equal to Ib1 times a list of ones, one per node
# significance: every node gets the same input bias (bias3)
transponder_bias1  = Ib1*np.ones(N_transponders)
# [orig l.115] set transponder_bias2 equal to Ib2 times a list of ones, one per node
# significance: every node gets the same output bias (bias4)
transponder_bias2  = Ib2*np.ones(N_transponders)

# [orig l.117] IF any --ib1 flags were given
if args.ib1:
  # [orig l.118] FOR each --ib1 setting (a pair of numbers)
  for setting in args.ib1:
    # [orig l.120] set node number (int of the first number) of transponder_bias1 equal to the second number times
    #              1e-6
    # significance: the exception: that node's input bias, uA to A
    transponder_bias1[int(setting[0])] = setting[1]*1.0e-6
# [orig l.121] IF any --ib2 flags were given
if args.ib2:
  # [orig l.122] FOR each --ib2 setting (a pair of numbers)
  for setting in args.ib2:
    # [orig l.123] set node number (int of the first number) of transponder_bias2 equal to the second number times
    #              1e-6
    # significance: the exception: that node's output bias, uA to A
    transponder_bias2[int(setting[0])] = setting[1] *1.0e-6

# [orig l.126] set transponder_current_divider equal to args.split
# significance: the split fraction. Used in ONE place only: RHS step 3, orig l.210, to set the pulse size sent to the
#               next gates
transponder_current_divider = args.split

# [orig l.129] set transponder_Rnw_input equal to a list of N_transponders zeros
# significance: U3's resistance R_nw1(t) for each node: 0 = superconducting. RHS updates it every step
transponder_Rnw_input = np.zeros(N_transponders)
# [orig l.130] set transponder_RT_loop equal to RT0 times a list of ones, one per node
# significance: loop 3's resistance R_T(t) for each node: R2, plus U2 while it is switched. RHS updates it every step
transponder_RT_loop   = RT0*np.ones(N_transponders)
# [orig l.131] set transponder_input_gate_current equal to a list of zeros
# significance: a record of the gate current, for plotting only
transponder_input_gate_current = np.zeros(N_transponders)
# [orig l.132] set transponder_source_current equal to a list of zeros
# significance: a record of (U3 current + gate current), for plotting only
transponder_source_current = np.zeros(N_transponders)

# [orig l.134] set init_input_current equal to transponder_bias1
# significance: each U3 starts out carrying its whole bias current
init_input_current  = transponder_bias1
# [orig l.135] set init_loop_currents equal to a list of zeros
# significance: the storage loops start empty
init_loop_currents  = np.zeros(N_transponders)


# [orig l.138] set transponder_initial_state equal to an empty list of 2 times N_transponders slots (16)
# significance: the starting values given to the solver: 2 numbers per node
transponder_initial_state =  np.empty((2*N_transponders), dtype=transponder_RT_loop.dtype)
# [orig l.139] set every other slot, starting at the first, equal to init_input_current
# significance: even slots = U3 channel current i_nw = the bias
transponder_initial_state[0::2] = init_input_current
# [orig l.140] set every other slot, starting at the second, equal to init_loop_currents
# significance: odd slots = storage-loop current i_loop = 0
transponder_initial_state[1::2] = init_loop_currents


# ---------------- STEP 7: flip the wiring table to 'who sends to me?' ----------------
# [orig l.144] set input_connections equal to an empty list for each node
# significance: will hold, for each receiver, the nodes that send to it
input_connections = [[] for _ in range(len(input_states))] 
# [orig l.145] set input_delays equal to delay transposed (rows and columns swapped)
# significance: now row = receiver, column = sender
input_delays = delay.T
# [orig l.146] FOR each row of the transposed connectivity table (i = row number, arow = the row)
# significance: each row = one receiver
for i,arow in enumerate(connectivity.T):
  # [orig l.147] FOR each entry in that row (j = position, acol = the value)
  # significance: each entry = one possible sender
  for j,acol in enumerate(arow):
    # [orig l.148] IF the entry equals 1
    # significance: a link exists from j to i
    if acol == 1:
      # [orig l.149] add j to the list input_connections[i]
      # significance: remember that node j sends to node i
      input_connections[i].append(j)


# [orig l.154] set output_states equal to a list of N_transponders zeros
# significance: same as orig l.101, done a second time (harmless)
output_states = np.zeros(N_transponders)
# [orig l.156] set output_click_times equal to an empty list for each node
# significance: same as orig l.104, done a second time (harmless)
output_click_times = [[] for _ in range(len(input_states))]



# ---------------- STEP 8: RHS, the engine ----------------
# [orig l.159] define a function RHS. Its inputs are t and x
# significance: THE ENGINE. The solver calls it at every tiny time step with the time t and all 16 currents x, and it
#               returns how fast each current is changing
def RHS(t, x):
  # [orig l.162] set nw_currents equal to every other entry of x, starting at the first
  # significance: the U3 channel current of every node
  nw_currents   = x[::2]  
  # [orig l.163] set loop_currents equal to every other entry of x, starting at the second
  # significance: the storage-loop current of every node
  loop_currents = x[1::2]

  # [orig l.165] set eps equal to 1e-11
  # significance: 10 ps: 'close enough' when matching times
  eps = 1.0e-11

    # ===== RHS step 1: the sensors fire (orig l.167-175). The outside world pokes the network =====
  # [orig l.167] FOR each sensor (n = its number, clicks = its list of scheduled click times)
  for n, clicks in enumerate(sensor_pulses):
    # [orig l.168] FOR each scheduled click time of that sensor
    for i_click, sensor_click_t in enumerate(clicks):  
      # [orig l.169] IF the absolute value of (t minus the click time) is at most eps
      # significance: it is now (within 10 ps) the time this sensor clicks
      if( abs(t -sensor_click_t) <= eps ) :
        # [orig l.170] print the sensor number, the click number and the times
        # significance: debug output
        print(f"n = {n} i_click = {i_click} time {t} vs {sensor_click_t}")
        # [orig l.171] set the state of sensor n to 1
        # significance: marked as clicked
        output_states[n] = 1
        # [orig l.174] add t to the list of click times of sensor n
        # significance: RECORD THE CLICK TIME. Every node listening to this sensor will find it in RHS step 3
        output_click_times[n].append(t)
        # [orig l.175] remove this click from the sensor's schedule
        # significance: so it only fires once
        sensor_pulses[n].pop(i_click)


    # ===== RHS step 2: output nTron U2 (orig l.177-197). Did each transponder fire? =====
  # [orig l.178] FOR each node (i_out = its number)
  for i_out, output_state in enumerate(output_states):
    # [orig l.179] IF the state of node i_out's U2 equals 1
    # significance: U2 is switched (it fired recently)
    if output_states[i_out] == 1:
      # [orig l.183] set tlast equal to the last entry in node i_out's list of spike times
      # significance: when it last fired
      tlast                       = output_click_times[i_out][len(output_click_times[i_out])-1]
      # [orig l.185] set node i_out's loop resistance equal to RT0 plus Rnorm times LinearPulse(t, starting at tlast,
      #              width 2 ns)
      # significance: while U2 is switched, loop 3's resistance is R2 plus a 1000 ohm pulse. That dumps the stored
      #               loop current
      transponder_RT_loop[i_out]  = RT0 + Rnorm*LinearPulse(t,t0=tlast,width=2.0e-9)
      # [orig l.187] IF (t minus tlast) is greater than ref_period
      # significance: 3 ns have passed since it fired
      if t-tlast > ref_period:
        # [orig l.188] set node i_out's U2 state to 0
        # significance: U2 is ready to fire again
        output_states[i_out] = 0
        # [orig l.189] print 'resetting output ntron'
        # significance: debug output
        print("resetting output ntron")
    # [orig l.190] IF the state of node i_out's U2 equals 0
    # significance: U2 is ready
    if output_states[i_out] == 0:
      # [orig l.191] set node i_out's loop resistance equal to RT0
      # significance: loop 3's resistance is just R2
      transponder_RT_loop[i_out] = RT0
      # [orig l.193] IF (node i_out's storage-loop current plus its output bias) is greater than Inw2_sw
      # significance: THE FIRING TEST. Storage-loop current (U2's gate) + bias4 beats U2's threshold
      if loop_currents[i_out] + transponder_bias2[i_out] > Inw2_sw: 
        # [orig l.195] print the node number and the time
        # significance: debug output
        print("output {}: click_t = {}".format(i_out, t))
        # [orig l.196] set node i_out's U2 state to 1
        # significance: U2 switches
        output_states[i_out] = 1
        # [orig l.197] add t to node i_out's list of spike times
        # significance: RECORD THE SPIKE TIME. This is the moment current leaves the transponder
        output_click_times[i_out].append(t)


    # ===== RHS step 3: input nTron U3 (orig l.199-236). Did enough arrive from upstream? =====
  # [orig l.201] FOR each node (i_in = its number)
  for i_in, input_state in enumerate(input_states):
    # [orig l.202] IF the state of node i_in's U3 equals 0
    # significance: U3 is ready
    if input_states[i_in] == 0:
      # [orig l.203] set node i_in's U3 resistance equal to 0
      # significance: superconducting
      transponder_Rnw_input[i_in] = 0
      # [orig l.205] set ntron_gate_current equal to 0
      # significance: start adding up the gate current from zero
      ntron_gate_current = 0
      # [orig l.206] set node i_in's recorded gate current equal to 0
      # significance: plotting record
      transponder_input_gate_current[i_in] = 0
      # [orig l.207] FOR each node j_in that sends to node i_in
      for j_in in input_connections[i_in]:
        # [orig l.208] set n_split equal to the sum of row j_in of connectivity
        # significance: how many nodes the sender feeds (its fan-out)
        n_split = np.sum(connectivity[j_in])
        # [orig l.209] set b2 equal to node j_in's output bias
        # significance: the sender's bias4
        b2  = transponder_bias2[j_in]
        # [orig l.210] set amp equal to b2 times transponder_current_divider divided by n_split
        # significance: THE PULSE SIZE from this sender = Ib2 x split / fan-out. The only place split is used
        amp = b2*transponder_current_divider/n_split
        # [orig l.212] FOR each spike time of node j_in
        # significance: every spike it has ever fired
        for click_t in output_click_times[j_in]:
          # [orig l.213] set click_delay equal to the delay from j_in to i_in
          # significance: the link delay (the delay box in the schematic)
          click_delay         = input_delays[i_in][j_in]
          # [orig l.214] set click_arrival equal to click_t plus click_delay
          # significance: when that spike arrives at node i_in
          click_arrival       = click_t  + click_delay
          # [orig l.215] add amp times OutputPulse(t, starting at click_arrival) to ntron_gate_current
          # significance: this pulse's current right now is added to the gate current (loop 1). Pulses from several
          #               senders add up
          ntron_gate_current += amp*OutputPulse(t, t0=click_arrival)
      # [orig l.218] set node i_in's recorded source current equal to its U3 current plus ntron_gate_current
      # significance: plotting record
      transponder_source_current[i_in]     = nw_currents[i_in] + ntron_gate_current
      # [orig l.219] set node i_in's recorded gate current equal to ntron_gate_current
      # significance: plotting record
      transponder_input_gate_current[i_in] = ntron_gate_current
      # [orig l.220] IF (node i_in's U3 current plus ntron_gate_current) is greater than Inw1_sw
      # significance: THE CLICK TEST. U3 channel current + gate current beats U3's threshold
      if nw_currents[i_in] + ntron_gate_current > Inw1_sw: 
        # [orig l.222] set node i_in's U3 state to 1
        # significance: U3 clicks
        input_states[i_in] = 1
        # [orig l.223] add t to node i_in's list of input-click times
        # significance: record the click time
        input_click_times[i_in].append(t)
    # [orig l.226] IF the state of node i_in's U3 equals 1
    # significance: U3 is clicked (resistive)
    if input_states[i_in] == 1:
      # [orig l.228] set latest_input_click equal to the last entry in node i_in's list of input-click times
      # significance: when it clicked
      latest_input_click = input_click_times[i_in][len(input_click_times[i_in])-1]
      # [orig l.229] IF (t minus latest_input_click) is greater than ref_period
      # significance: 3 ns have passed since the click
      if t-latest_input_click > ref_period:
        # [orig l.231] print 'input reset'
        # significance: debug output
        print("input reset")
        # [orig l.232] set node i_in's U3 state to 0
        # significance: U3 is ready again
        input_states[i_in] = 0
      # [orig l.234] OTHERWISE
      # significance: less than 3 ns since the click
      else: 
        # [orig l.236] set node i_in's U3 resistance equal to Rnorm times LinearPulse(t, starting at
        #              latest_input_click, width 1 ns)
        # significance: U3 is resistive: a 1000 ohm trapezoid about 2.5 ns long. This pushes current out of U3 into
        #               the other branches
        transponder_Rnw_input[i_in] = Rnorm*LinearPulse(t,t0=latest_input_click, width=1.0e-9)

    # ===== RHS step 4: circuit physics (orig l.237-250). Where does the current go? =====
  # [orig l.237] set y equal to an empty list
  # significance: will hold the rate of change of every current
  y = []
  # [orig l.238] FOR each node (i_node = its number, loop_cur = its storage-loop current)
  for i_node,loop_cur in enumerate(loop_currents):
    # [orig l.239] set i1 equal to node i_node's input bias
    # significance: bias3 flowing in
    i1  = transponder_bias1[i_node]
    # [orig l.240] set inw equal to node i_node's U3 current
    inw = nw_currents[i_node]
    # [orig l.241] set iloop equal to loop_cur
    # significance: the storage-loop current
    iloop = loop_cur
    # [orig l.242] set Rnw equal to node i_node's U3 resistance
    # significance: from RHS step 3
    Rnw =  transponder_Rnw_input[i_node]
    # [orig l.243] set RT equal to node i_node's loop resistance
    # significance: from RHS step 2
    RT  = transponder_RT_loop[i_node]
    # [orig l.246] set y1 equal to ( minus inw*Ls1*Rnw, minus i1*LT*Rs1, plus iloop*LT*Rs1, plus inw*LT*(Rnw + Rs1),
    #              plus iloop*Ls1*RT ), all divided by ( Lnw*(Ls1 minus LT) minus Ls1*LT )
    # significance: y1 = d(i_nw)/dt, how fast U3's current is changing (Kirchhoff's laws with the inductors). Caution:
    #               the Lnw*Ls1 term has the opposite sign from the circuit as drawn (about 5% on time constants);
    #               worth checking with Whit
    y1 = (-inw*Ls1*Rnw - i1*LT*Rs1 + iloop*LT*Rs1 + inw*LT*(Rnw + Rs1) + iloop*Ls1*RT)/(Lnw*(Ls1 - LT) - Ls1*LT)
    # [orig l.247] set y2 equal to ( minus inw*Ls1*Rnw, minus i1*Lnw*Rs1, plus iloop*Lnw*Rs1, plus inw*Lnw*Rs1, plus
    #              iloop*Lnw*RT, plus iloop*Ls1*RT ), all divided by ( Lnw*(Ls1 minus LT) minus Ls1*LT )
    # significance: y2 = d(i_loop)/dt, how fast the storage-loop current is changing
    y2 = (-inw*Ls1*Rnw - i1*Lnw*Rs1 + iloop*Lnw*Rs1 + inw*Lnw*Rs1 + iloop*Lnw*RT + iloop*Ls1*RT)/(Lnw*(Ls1 - LT) - Ls1*LT)
    # [orig l.248] add y1 to the list y
    y.append(y1)
    # [orig l.249] add y2 to the list y
    y.append(y2)
  # [orig l.250] RETURN the list y
  # significance: hands all 16 rates back to the solver, which takes the next step and calls RHS again
  return y




# ---------------- STEP 9: main(), run the simulation and make the original plot ----------------
# [orig l.254] define a function main that returns a whole number
def main() -> int:
  """ Prototype network simulation"""
  # [orig l.256] print the delay table
  print(delay)
  


  # [orig l.271] set t_pts equal to 1000 evenly spaced times from 0 to 100 times 1e-9 seconds
  # significance: the times at which results are saved (0 to 100 ns)
  t_pts = np.linspace(0, 100*1.0e-9, num=1000)

  # [orig l.273] print the split fraction
  print(f"shunt/output current splilt: {transponder_current_divider}")

  # [orig l.275] print 'Starting simulation'
  print("Starting simulation")
  # [orig l.276] set solution equal to the result of solve_ivp, which solves RHS from time 0 to 100 ns, starting from
  #              transponder_initial_state, saving at t_pts, with steps no larger than 1 ps
  # significance: RUNS THE SIMULATION (about a minute). The 1 ps step limit stops it skipping over sub-nanosecond
  #               switching events
  solution = solve_ivp(RHS, (0, 100 * 1.0e-9), transponder_initial_state, t_eval= t_pts, max_step = 1.0e-12)

  # [orig l.278] create a figure with 6 stacked panels, called fig, with the panels in axs
  # significance: the original plot: for t1, t2, t3 the U3 current (labelled 'gate') and the loop current (red)
  fig, axs = plt.subplots(6)
  # [orig l.280] in panel 0, plot solution row 4 divided by 1e-6 against solution.t divided by 1e-9, labelled 'gate 2'
  # significance: row 2n of the solution = i_nw of node n; row 4 = node 2 = t1. Microamps vs ns
  axs[0].plot(solution.t/1.0e-9, solution.y[4]/1.0e-6, label='gate 2')
  # [orig l.281] in panel 1, plot solution row 5 in microamps against time in ns, in red, labelled 'loop 2'
  # significance: row 2n+1 = i_loop of node n
  axs[1].plot(solution.t/1.0e-9, solution.y[5]/1.0e-6, 'r',label='loop 2')
  # [orig l.282] in panel 2, plot solution row 6 in microamps against time in ns, labelled 'gate 3'
  # significance: i_nw of t2
  axs[2].plot(solution.t/1.0e-9, solution.y[6]/1.0e-6, label='gate 3')
  # [orig l.283] in panel 3, plot solution row 7 in microamps against time in ns, in red, labelled 'loop 3'
  # significance: i_loop of t2
  axs[3].plot(solution.t/1.0e-9, solution.y[7]/1.0e-6, 'r',label='loop 3')
  # [orig l.284] in panel 4, plot solution row 8 in microamps against time in ns, labelled 'gate 4'
  # significance: i_nw of t3
  axs[4].plot(solution.t/1.0e-9, solution.y[8]/1.0e-6, label='gate 4')
  # [orig l.285] in panel 5, plot solution row 9 in microamps against time in ns, in red, labelled 'loop 4'
  # significance: i_loop of t3
  axs[5].plot(solution.t/1.0e-9, solution.y[9]/1.0e-6, 'r',label='loop 4')

  # [orig l.291] in panel 0, write text showing dt, the input biases and the output biases
  # significance: settings printed on the figure
  axs[0].annotate(f"dt={dt//1.0e-9}, \nb1={transponder_bias1/1.0e-6}, \nb2={transponder_bias2/1.0e-6}", xycoords="subfigure points",xy=(1,1))

  # [orig l.293] in panel 0, show the legend
  axs[0].legend()
  # [orig l.294] in panel 1, show the legend
  axs[1].legend()
  # [orig l.295] in panel 2, show the legend
  axs[2].legend()
  # [orig l.296] in panel 3, show the legend
  axs[3].legend()
  # [orig l.297] in panel 4, show the legend
  axs[4].legend()
  # [orig l.298] in panel 5, show the legend
  axs[5].legend()

  # [orig l.300] FOR each panel
  for ax in axs.flat:
    # [orig l.301] set its x label to 'time [ns]' and its y label to 'current [uA]'
    ax.set(xlabel='time [ns]',ylabel='current [uA]')

  # [orig l.304] FOR each panel
  for ax in axs.flat:
    # [orig l.305] hide the repeated axis labels on inner panels
    ax.label_outer()
  # [orig l.307] save the figure as ../results/<output>.png at 200 dpi
  plt.savefig(f"../results/{args.output}.png", bbox_inches='tight',dpi=200)
  # [orig l.308] save the figure as ../results/<output>.pdf
  plt.savefig(f"../results/{args.output}.pdf", format="pdf", bbox_inches='tight',dpi=200)

  # [orig l.311] print the solver's status
  # significance: 0 means it finished without error
  print(f"solution status {solution.status}")


    # ---------------- STEP 10 [new]: rebuild the gate current, then the three plots you asked for ----------------
  # [new] set t equal to the solver's time points (in seconds)
  t   = solution.t
  # [new] set tns equal to t divided by 1e-9
  # significance: the same times in ns
  tns = t/1.0e-9
  # [new] set names equal to the list 'i1', 'i2', 't1', ... 't6'
  # significance: labels for the 8 nodes
  names = ['i1', 'i2'] + [f't{k}' for k in range(1, N_transponders-1)]
  # [new] set gate_parts equal to an empty dictionary for each node
  # significance: gate_parts[i][j] will be the current node i receives from node j. RHS computed these but threw them
  #               away
  gate_parts = [dict() for _ in range(N_transponders)]
  # [new] set gate_current equal to a table of zeros, one row per node and one column per saved time
  # significance: the total gate current of each node
  gate_current = np.zeros((N_transponders, len(t)))
  # [new] FOR each node i_in
  # significance: each receiver
  for i_in in range(N_transponders):
    # [new] FOR each node j_in that sends to i_in
    for j_in in input_connections[i_in]:
      # [new] set amp equal to node j_in's output bias times transponder_current_divider divided by the sum of row
      #       j_in of connectivity
      # significance: pulse size = sender's Ib2 x split / fan-out (same as orig l.208-210)
      amp  = transponder_bias2[j_in]*transponder_current_divider/np.sum(connectivity[j_in])
      # [new] set part equal to a list of zeros, one per saved time
      # significance: this sender's contribution over time
      part = np.zeros(len(t))
      # [new] FOR each spike time of node j_in
      for click_t in output_click_times[j_in]:
        # [new] set arrival equal to click_t plus the delay from j_in to i_in
        # significance: when that spike reaches i_in (orig l.213-214)
        arrival = click_t + input_delays[i_in][j_in]
        # [new] add to part, at every saved time tk, amp times OutputPulse(tk, starting at arrival) if tk is at least
        #       click_t, otherwise 0
        # significance: the pulse only counts after the spike happened, as RHS cannot see the future
        part += np.array([amp*OutputPulse(tk, t0=arrival) if tk >= click_t else 0.0 for tk in t])
      # [new] store part as the contribution of j_in to i_in
      gate_parts[i_in][j_in] = part
      # [new] add part to node i_in's total gate current
      gate_current[i_in] += part

  # [new] print 'output (spike) click times [ns]:'
  print("output (spike) click times [ns]:")
  # [new] FOR each node n
  for n in range(N_transponders):
    # [new] print the node's name and its spike times in ns, rounded to 2 decimals
    print(f"  {names[n]}: {np.round(np.array(output_click_times[n])/1.0e-9, 2)}")


    # ---- SPIKE RASTER ----
  # [new] create a new figure 8 by 3.5 inches
  # significance: SPIKE RASTER: one row per node, a black tick for each spike, a red triangle for each input click
  fig, ax = plt.subplots(figsize=(8, 3.5))
  # [new] FOR each node n
  for n in range(N_transponders):
    # [new] draw a vertical black line from n minus 0.4 to n plus 0.4 at each of node n's spike times in ns, labelled
    #       'output click (spike)' only for node 0
    # significance: one black tick per spike
    ax.vlines(np.array(output_click_times[n])/1.0e-9, n-0.4, n+0.4, color='k', lw=2,
              label='output click (spike)' if n == 0 else None)
    # [new] IF node n has any input clicks
    if input_click_times[n]:
      # [new] plot a red downward triangle just under row n at each of node n's input-click times
      # significance: one red triangle per input click
      ax.plot(np.array(input_click_times[n])/1.0e-9, [n-0.45]*len(input_click_times[n]), 'v', color='tab:red', ms=4,
              label='input (gate) click' if not any(input_click_times[:n]) else None)
  # [new] FOR y equal to 1.5 and then 4.5
  for y in (1.5, 4.5):
    # [new] draw a light grey dashed horizontal line at y
    # significance: separates sensors | t1-t3 | t4-t6
    ax.axhline(y, color='0.8', ls='--', lw=0.8)
  # [new] label the rows with the node names
  ax.set_yticks(range(N_transponders), names)
  # [new] set the vertical range from N_transponders minus 0.5 down to minus 0.5
  # significance: puts i1 at the top
  ax.set_ylim(N_transponders-0.5, -0.5)
  # [new] set the x range to 0 up to the last time, the x label to 'time [ns]' and the title to 'spike raster'
  ax.set(xlim=(0, tns[-1]), xlabel='time [ns]', title='spike raster')
  # [new] show the legend at the upper right, small, without a frame
  ax.legend(loc='upper right', fontsize=8, frameon=False)
  # [new] save this figure as ../results/<output>_raster.png at 200 dpi
  plt.savefig(f"../results/{args.output}_raster.png", bbox_inches='tight', dpi=200)
  # [new] save this figure as ../results/<output>_raster.pdf
  plt.savefig(f"../results/{args.output}_raster.pdf", bbox_inches='tight')


    # ---- GATE-CURRENT TRACE ----
  # [new] set n equal to the --trace_node flag
  # significance: GATE-CURRENT TRACE for one node: do the incoming pulses add up above threshold?
  n = args.trace_node
  # [new] set inw equal to row 2 times n of the solution
  # significance: node n's U3 channel current over time
  inw = solution.y[2*n]
  # [new] create a new figure with 2 stacked panels sharing the x axis, 8 by 5 inches
  fig, axs2 = plt.subplots(2, sharex=True, figsize=(8, 5))
  # [new] FOR each sender j_in (and its pulse part) that feeds node n
  for j_in, part in gate_parts[n].items():
    # [new] in the top panel, plot that sender's pulse in microamps as a dashed line, labelled 'from <sender>'
    # significance: TOP panel: each sender's pulse
    axs2[0].plot(tns, part/1.0e-6, '--', label=f'from {names[j_in]}')
  # [new] in the top panel, plot node n's total gate current in microamps as a black line
  # significance: the sum of the pulses
  axs2[0].plot(tns, gate_current[n]/1.0e-6, 'k', lw=1.5, label='summed gate current')
  # [new] in the top panel, draw a red dotted horizontal line at (Inw1_sw minus node n's input bias) in microamps
  # significance: the gate current needed to make U3 click
  axs2[0].axhline((Inw1_sw - transponder_bias1[n])/1.0e-6, color='r', ls=':', label='needed to switch (I1sw - Ib1)')
  # [new] in the top panel, set the y label and the title
  axs2[0].set(ylabel='gate current [uA]', title=f'{names[n]}: gate current')
  # [new] in the top panel, show the legend
  axs2[0].legend(fontsize=8, frameon=False)
  # [new] in the bottom panel, plot inw in microamps as a grey line, labelled 'channel current i_nw'
  # significance: BOTTOM panel: U3's own current
  axs2[1].plot(tns, inw/1.0e-6, color='0.5', label='channel current i_nw')
  # [new] in the bottom panel, plot (inw plus node n's gate current) in microamps as a black line
  # significance: what the click test compares with the threshold (orig l.220)
  axs2[1].plot(tns, (inw + gate_current[n])/1.0e-6, 'k', label='i_nw + gate')
  # [new] in the bottom panel, draw a red dotted horizontal line at Inw1_sw in microamps
  # significance: U3's threshold
  axs2[1].axhline(Inw1_sw/1.0e-6, color='r', ls=':', label='I1sw (threshold)')
  # [new] FOR each input-click time tc of node n (k = its count)
  for k, tc in enumerate(input_click_times[n]):
    # [new] in the bottom panel, draw a faint red vertical line at tc
    # significance: marks each click
    axs2[1].axvline(tc/1.0e-9, color='r', alpha=0.4, label='input click' if k == 0 else None)
  # [new] in the bottom panel, set the x and y labels
  axs2[1].set(xlabel='time [ns]', ylabel='current [uA]')
  # [new] in the bottom panel, show the legend
  axs2[1].legend(fontsize=8, frameon=False)
  # [new] save this figure as ../results/<output>_gate_trace_<node>.png at 200 dpi
  plt.savefig(f"../results/{args.output}_gate_trace_{names[n]}.png", bbox_inches='tight', dpi=200)
  # [new] save this figure as ../results/<output>_gate_trace_<node>.pdf
  plt.savefig(f"../results/{args.output}_gate_trace_{names[n]}.pdf", bbox_inches='tight')


    # ---- GRADIENT RASTER ----
  # [new] set gmax equal to the largest value in gate_current if that is above 0, otherwise 1
  # significance: GRADIENT RASTER: the largest gate current anywhere in the run (avoids dividing by 0)
  gmax  = gate_current.max() if gate_current.max() > 0 else 1.0
  # [new] set shade equal to gate_current divided by gmax
  # significance: every value scaled to between 0 (no current) and 1 (the maximum)
  shade = gate_current/gmax
  # [new] FOR each node n
  for n in range(N_transponders):
    # [new] FOR each input-click time tc of node n
    for tc in input_click_times[n]:
      # [new] set shade to 0 for node n at every time after tc up to tc + ref_period
      # significance: white right after a click: the nanowire is recovering
      shade[n][(t > tc) & (t <= tc + ref_period)] = 0
  # [new] create a new figure 8 by 3.5 inches
  fig, ax = plt.subplots(figsize=(8, 3.5))
  # [new] draw shade as a grey-scale image (0 = white, 1 = black) with time along x and one row per node
  # significance: the gradient
  im = ax.imshow(shade, aspect='auto', cmap='Greys', vmin=0, vmax=1, interpolation='nearest',
                 extent=(tns[0], tns[-1], N_transponders-0.5, -0.5))
  # [new] FOR each node n
  for n in range(N_transponders):
    # [new] draw a thick red vertical line across row n at each of its spike times
    # significance: solid red = spike
    ax.vlines(np.array(output_click_times[n])/1.0e-9, n-0.5, n+0.5, color='red', lw=1.8,
              label='output click (spike)' if n == 0 else None)
    # [new] draw a thin dotted red vertical line across row n at each of its input-click times
    # significance: dotted red = input click
    ax.vlines(np.array(input_click_times[n])/1.0e-9, n-0.5, n+0.5, color='red', lw=0.8, ls=':',
              label='input (gate) click' if n == 2 else None)
  # [new] show the legend below the plot in 2 columns
  ax.legend(loc='upper center', fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.15))
  # [new] FOR y equal to 1.5 and then 4.5
  for y in (1.5, 4.5):
    # [new] draw a light grey dashed horizontal line at y
    # significance: separators
    ax.axhline(y, color='0.8', ls='--', lw=0.8)
  # [new] label the rows with the node names
  ax.set_yticks(range(N_transponders), names)
  # [new] set the x label to 'time [ns]' and the title to 'gradient raster: gate current (white = 0, black = max)'
  ax.set(xlabel='time [ns]', title='gradient raster: gate current (white = 0, black = max)')
  # [new] add a colour bar labelled 'gate current / <gmax> uA'
  # significance: reads shade back as a fraction of the largest gate current
  fig.colorbar(im, ax=ax, label=f'gate current / {gmax/1.0e-6:.2f} uA', pad=0.01)
  # [new] save this figure as ../results/<output>_gradient_raster.png at 200 dpi
  plt.savefig(f"../results/{args.output}_gradient_raster.png", bbox_inches='tight', dpi=200)
  # [new] save this figure as ../results/<output>_gradient_raster.pdf
  plt.savefig(f"../results/{args.output}_gradient_raster.pdf", bbox_inches='tight')

  # [orig l.314] RETURN 0
  # significance: main finished successfully
  return 0

# [orig l.316] IF this file is being run directly (not imported)
if __name__ == '__main__':
  # [orig l.317] run main() and exit the program with its return value
  sys.exit(main())
