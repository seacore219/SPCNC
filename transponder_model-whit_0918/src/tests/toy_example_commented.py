# ======================================================================================================
# COMMENTED COPY of toy_example.py  (same code, comments only; the code is checked to be identical)
#
# Comment key. Above each statement:
#   [orig l.N]      N = line number in toy_example.py on `main` (Whit's original, 317 lines); [new] = code that
#                   is not in the original (recurrent wiring, raster, gate trace, gradient raster).
#   how:            how the statement works as Python/numpy: what each function, operator or symbol does, so you
#                   could write it yourself. Left out when the line is self-explanatory.
#   significance:   why the line matters for the model / which part of the schematic it is
#                   (single_transponder_simple.pdf: U3 = input nTron (Q1), U2 = output nTron (Q2),
#                   bias3 = I_b1, bias4 = I_b2, loop 1 = input/gate loop, loop 2 = Q1 shunt (L1, R3),
#                   loop 3 = storage loop (Lb1, R2), loop 4 = output loop (L2, R1)).
# Comments inside the delay table are yours and are kept.
#
# How the program runs: (1) read flags, (2) set up the network tables and starting currents,
# (3) RHS() says how fast every current changes at an instant, (4) main() hands RHS to a solver that
# steps through 100 ns, (5) plots.
# ======================================================================================================

# ---------------- STEP 1: libraries ----------------
# [orig l.1] how: `import X as Y` loads the library X and lets you call it by the short name Y: from now on
#                 np.array(...) means numpy.array(...).
# significance: numpy gives us arrays (tables of numbers) and maths that works on a whole table at once.
import numpy as np
# [orig l.2] how: `from A import B` loads only the function B from the module A, so it can be called as solve_ivp(...)
#                 with no prefix.
# significance: the solver that steps the circuit equations forward in time (used in main).
from scipy.integrate import solve_ivp
# [orig l.3] significance: plotting library.
import matplotlib.pyplot as plt
# [orig l.4] significance: only used on the last line, to hand an exit status back to the operating system.
import sys
# [orig l.5] significance: gives math.exp(), used in OutputPulse.
import math
# [orig l.6] significance: reads the command-line flags (--Ib1init and so on).
import argparse
# [orig l.7] significance: only used when you run with --generator.
import spcnc.tof_input_generator as tof_gen

# [orig l.9] how: ArgumentParser(...) creates an empty flag-reader object. prog and description are only text shown by
#                 `--help`; the flags themselves are registered one at a time below with add_argument.
# significance: nothing is read from the command line yet.
parser = argparse.ArgumentParser(
                    prog='spcnc_toy_example',
                    description='prototype spiking network simulation',
                    epilog='')


# ---------------- STEP 2: the two pulse shapes ----------------
# [orig l.14] how: `def name(inputs):` defines a reusable function; the indented lines below are its body and run each
#                  time it is called. An input written `t0 = 5.0e-9` has a default value, used when the caller leaves
#                  that input out. 5.0e-9 is scientific notation for 5.0 x 10^-9, i.e. 5 ns written in seconds.
# significance: a trapezoid between 0 and 1: t = now, t0 = when it starts, width = how long it stays at 1, rise / fall
#               = ramp times. It is NOT a resistance by itself: the callers multiply it by Rnorm (see the `return 0`
#               note below).
def LinearPulse(t, t0 = 5.0e-9, width=0.5e-9, rise=1.0e-9, fall=0.5e-9):
  # [orig l.15] how: `if condition:` runs the indented lines below it only when the condition is true. `or` is true
  #                  when at least one side is true, so this means: now is before the start (t-t0 < 0) or more than
  #                  rise + width + fall after it (the pulse is over).
  # significance: outside the pulse.
  if (t-t0 < 0) or (t-t0 > width+fall+rise): 
    # [orig l.16] significance: `return` ends the function and hands the value back. LinearPulse returns only a plain
    #                           number from 0 to 1; it knows nothing about ohms. The two callers turn it into a
    #                           resistance by multiplying by Rnorm (1000 ohm): U2 at orig l.185 (RT = RT0 +
    #                           Rnorm*LinearPulse(...)) and U3 at orig l.236 (Rnw = Rnorm*LinearPulse(...)). So 0 here
    #                           = 0 ohm = superconducting. Those resistances (Rnw, RT; orig l.242-243) enter the
    #                           circuit equations y1/y2 (orig l.246-247), where Rnw = 0 leaves U3 a plain wire and Rnw
    #                           = 1000 forces its current into the shunt and the storage loop.
    return 0
  # [orig l.17] how: `and` is true only when both sides are true.
  # significance: rising edge: the pulse has started but not yet reached full height.
  if ((t-t0) > 0) and ((t - t0) < rise):
    # [orig l.19] how: `name = expression` evaluates the right-hand side and stores it under the name, so the next
    #                  line can use it.
    # significance: time elapsed since the pulse started.
    delta_t = t-t0
    # [orig l.21] how: elapsed time divided by the rise time is a fraction: 0 at the start, 1 once `rise` has elapsed.
    # significance: a straight-line ramp 0 -> 1, so the resistance climbs from 0 to Rnorm over `rise`.
    return delta_t/rise 
  # [orig l.22] how: `<=` means 'less than or equal'. The window is from (width + rise) to (width + fall + rise) after
  #                  the start.
  # significance: falling edge.
  if (t-t0 > width+rise) and (t-t0 <= width+fall+rise):
    # [orig l.24] how: (t0+width+fall+rise) is the moment the pulse ends; subtracting t gives the time still
    #                  remaining.
    # significance: time left until the pulse ends.
    delta_t = (t0+width+fall+rise)-t
    # [orig l.26] how: time remaining divided by `fall`: 1 when exactly `fall` remains, 0 when none remains.
    # significance: a straight-line ramp 1 -> 0 (resistance back to 0).
    return (delta_t/fall)
  # [orig l.27] how: `else:` runs when none of the `if` conditions above were true.
  # significance: the flat top, between the end of the rise and the start of the fall. (Caution: `else` also catches
  #               the single instant t == t0, so at exactly the start time the function returns 1, not 0.)
  else: 
   # [orig l.28] significance: full height = Rnorm = 1000 ohm when multiplied as above.
   return 1

# [orig l.30] significance: the shape of the current one upstream spike delivers into a node's gate (loop 1). t0 =
#                           where it is about half way up, tau = decay time, mu0 = how sharp the rise is.
def OutputPulse(t, t0=1e-9, tau=1.0e-9, mu0=0.2e-9):
  # [orig l.33] how: math.exp(x) is e to the power x. `1.0 - 1.0/(math.exp(a) + 1.0)` equals 1/(1 + e^-a), an S-shaped
  #                  step: near 0 when a is very negative, 0.5 when a = 0, near 1 when a is large. Here a =
  #                  (t-t0)/mu0, so the step happens within about mu0 either side of t0. It is multiplied by
  #                  math.exp(-(t-t0)/tau), an exponential that decays with time constant tau.
  # significance: a smooth step up around t0 followed by a decay; the pulse peaks at about 0.61. Because the step is
  #               smooth it is already about 20% of its amplitude 0.35 ns BEFORE t0, which is why a node can click
  #               slightly before a spike's nominal arrival.
  return (1.0 - 1.0/(math.exp((t - t0)/mu0) + 1.0))*math.exp(- ((t - t0)/tau))

"""
Units: can we default to us? 

There are two input pulses of fixed shape.
This defines the input currents 
"""

# ---------------- STEP 3: the command-line flags (the knobs) ----------------
# [orig l.42] how: `add_argument('--name', ...)` registers a flag. `type=float` converts the typed text to a number;
#                  `default=` is the value used when the flag is not typed.
# significance: base name of the plot files written to ../results.
parser.add_argument("--output", help="output plots file name", default="toy_example" )
# [orig l.43] significance: input bias of every node (uA). Schematic: I_b1, bias3 into U3's drain (LTspice uses ~150
#                           uA).
parser.add_argument("--Ib1init", help="I_b1 initial value in uA", type=float, default=16 )
# [orig l.44] significance: output bias of every node (uA). Schematic: I_b2, bias4 into U2's drain (LTspice uses ~160
#                           uA).
parser.add_argument("--Ib2init", help="I_b1 initial value in uA", type=float, default=16 )
# [orig l.45] significance: switching current of U3 = the input threshold (uA).
parser.add_argument("--I1sw", help="input ntron switching current in uA", type=float, default=20 )
# [orig l.46] significance: switching current of U2 = the output threshold (uA).
parser.add_argument("--I2sw", help="output ntron switching current in uA", type=float, default=20 )
# [orig l.47] significance: fraction of a spike's current sent on to the next gates; the rest goes to the shunt.
#                           Schematic: the L2 / R1 division of loop 4.
parser.add_argument("--split", help="current splitting fraction", type=float, default=0.1 )
# [orig l.48] how: `action='store_true'` makes the flag an on/off switch: False if not typed, True if typed (no value
#                  follows it).
# significance: use the time-of-flight generator to pick the gap between the two sensor clicks.
parser.add_argument("--generator", help="use tof generator" , action='store_true' )
# [orig l.49] significance: gap in ns between the two sensor clicks. Schematic: pulser2 (U4) fires this long after
#                           pulser1 (U1).
parser.add_argument("--dt", help="delta t in ns, ignored if generator is used", type=float, default=5 )

# [orig l.51] how: `nargs=2` requires exactly two values after the flag. `action='append'` lets the flag be typed
#                  several times, collecting each pair into one list: `--ib1 2 13 --ib1 4 11.2` becomes [[2.0, 13.0],
#                  [4.0, 11.2]].
# significance: --ib1 NODE VALUE sets one node's input bias. NODE = row of the delay table (0 = i1, 1 = i2, 2 = t1 ...
#               7 = t6), VALUE in uA.
parser.add_argument('--ib1', nargs=2, action='append', type=float, help='set node [0] input bias to value [1]')
# [orig l.52] significance: same as --ib1 but for the output bias (bias4).
parser.add_argument('--ib2', nargs=2, action='append', type=float, help='set node [0] output bias to value [1]')
# [new] how: `type=int` converts the typed text to a whole number.
# significance: which node the gate-current trace is drawn for (4 = t3).
parser.add_argument('--trace_node', type=int, default=4, help='matrix index of node for the gate-current trace plot (0=i1, 1=i2, 2=t1 ... 7=t6); default t3')
# [orig l.54] how: reads what was typed after the program name, checks it against the registered flags, and returns an
#                  object `args` with one field per flag (args.I1sw, args.split, ...). Flags that were not typed hold
#                  their default.
# significance: from here on the flag values are available as args.<name>.
args = parser.parse_args()

# [orig l.56] how: prints the value; it is None if --ib1 was never typed.
# significance: debug output.
print(args.ib1)


# ---------------- STEP 4: convert flags to SI units and set the fixed circuit values ----------------
# [orig l.58] how: args.I1sw is the number typed after --I1sw (default 20). Multiplying by 1e-6 converts microamps to
#                  amps.
# significance: U3's threshold in amps.
Inw1_sw = args.I1sw*1.0e-6
# [orig l.59] how: same conversion, microamps to amps.
# significance: U2's threshold in amps.
Inw2_sw = args.I2sw*1.0e-6

# [orig l.61] how: same conversion.
# significance: I_b1 (bias3) in amps.
Ib1 = args.Ib1init*1.0e-6
# [orig l.62] how: same conversion.
# significance: I_b2 (bias4) in amps.
Ib2 = args.Ib2init*1.0e-6

# [orig l.65] how: 5 * 1.0e-9 is 5 x 10^-9: nanohenries written in henries.
# significance: 5 nH, inductance of the shunt around U3. Schematic: L1 (loop 2).
Ls1 = 5 * 1.0e-9
# [orig l.66] significance: 2 ohm, resistance of that shunt. Schematic: R3 (loop 2); it is 3 ohm in LTspice.
Rs1 = 2
# [orig l.67] significance: 150 nH, storage-loop inductance. Schematic: Lb1 (~148 nH, loop 3).
LT  = 150 * 1.0e-9
# [orig l.68] significance: 3 ohm, storage-loop resistance while U2 is not switched. Schematic: R2 (loop 3); it is 4
#                           ohm in LTspice.
RT0 = 3

# [orig l.70] significance: 10 nH, inductance of an nTron channel. Schematic: L_nw inside U3 and U2.
Lnw   = 10 * 1.0e-9
# [orig l.71] significance: 1000 ohm, an nTron channel's resistance while switched. Schematic: R_nw(t) of U3; for U2
#                           it is added onto R2. LinearPulse x Rnorm = the resistance.
Rnorm = 1000

# [orig l.73] significance: never used anywhere (the original's own comment says so).
recover_time1 = 3.0e-9
# [orig l.75] how: 0.003 * 1.0e-6 = 3 x 10^-9 s.
# significance: 3 ns rest time after a click before that nTron can click again. (The original comment says 'not
#               used??' but it IS used, at orig l.187 and l.229.)
ref_period = 0.003 *1.0e-6



# ---------------- STEP 5: the wiring of the network ----------------
# [orig l.79] significance: (the wiring table follows; rows = senders, columns = receivers; rows and columns 0-1 are
#                           the sensors i1, i2, then t1...t6)
#  first two columns are inputs   
##                       to: i1  i2  t1  t2  t3  t4  t5  t6    from:
# [orig l.81] how: np.array([[...], [...], ...]) turns a list of rows into a 2-D table, here 8 rows x 8 columns.
#                  Multiplying the whole table by 1.0e-9 multiplies every entry (ns -> seconds). Entry delay[r][c]
#                  means row r, column c, counting from 0.
# significance: THE NETWORK IN ONE TABLE. Entry [row][column] = delay in ns of the link from the row node to the
#               column node, 0 = no link. CHANGED from the original, which was feedforward only (t1->t3, t2->t3,
#               t3->t4; t4-t6 sent nothing).
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
# [orig l.92] how: np.ceil(x) rounds every entry of x UP to the next whole number (0 stays 0; 0.2 -> 1; 2.3 -> 3).
#                  np.minimum(a, b) compares a and b entry by entry and keeps the smaller; the single number 1 is
#                  automatically stretched to match the whole table. So: ceil turns every positive delay into a whole
#                  number >= 1, and minimum then caps that at 1.
# significance: a table with 1 where a link exists and 0 where there is none. (The delays here are ~1e-9 seconds, so
#               np.ceil alone already gives 1; np.minimum only matters if a delay were over 1 second. It is a safety
#               clamp.)
connectivity = np.minimum(1,np.ceil(delay)) # converts binarizes delay function to either 0 or 1 (connnection or no connection)

# [orig l.94] how: np.shape(table) gives (number of rows, number of columns).
# significance: prints (8, 8).
print(np.shape(delay))
# [orig l.95] how: len(table) is the number of rows.
# significance: prints 8.
print(len(delay))

# [orig l.97] significance: 8 nodes (2 sensors + 6 transponders). Used everywhere to size the arrays.
N_transponders = len(delay)


# ---------------- STEP 6: starting bookkeeping (nothing has fired yet) ----------------
# [orig l.100] how: np.zeros(N) makes an array of N zeros.
# significance: state of every U3: 0 = ready (superconducting), 1 = clicked. All start ready.
input_states = np.zeros(N_transponders)
# [orig l.101] significance: state of every U2 (same codes). All start ready.
output_states = np.zeros(N_transponders)

# [orig l.103] how: `[ [] for _ in range(N) ]` is a list comprehension: it evaluates `[]` once for each value 0..N-1
#                   and collects the results, giving N separate empty lists. (Writing [[]]*N would give N references
#                   to ONE list, so all nodes would share it.)
# significance: the history of when each node's U3 clicked.
input_click_times = [[] for _ in range(N_transponders)]
# [orig l.104] significance: the history of when each node fired (its spike times).
output_click_times = [[] for _ in range(N_transponders)]

# [orig l.106] significance: gap between the two sensor clicks, ns -> seconds.
dt = 1.0e-9*args.dt
# [orig l.107] how: args.generator is False unless --generator was typed.
if args.generator:
  # [orig l.108]
  print("using generator")
  # [orig l.109] how: calls a function from the generator module that returns an event (a dictionary of values).
  ev = tof_gen.generate_tof_event()
  # [orig l.110] how: ev['tof'] looks up the value stored under the key 'tof' in the dictionary.
  # significance: the generator overrides the gap between the sensor clicks.
  dt = ev["tof"]

# [orig l.112] how: a list of two lists; each inner list holds that sensor's scheduled click times (one each here).
# significance: i1 clicks at 1 ns, i2 at 1 ns + dt. Schematic: pulser1 (U1) and pulser2 (U4).
sensor_pulses = [[1.0e-9],[1.0e-9 + dt]]

# [orig l.114] how: np.ones(N) is N ones; multiplying by Ib1 gives N copies of Ib1, one per node.
# significance: every node gets the same input bias (bias3).
transponder_bias1  = Ib1*np.ones(N_transponders)
# [orig l.115] significance: every node gets the same output bias (bias4).
transponder_bias2  = Ib2*np.ones(N_transponders)

# [orig l.117] how: an empty or missing (None) value counts as false, so this runs only if at least one --ib1 was
#                   typed.
if args.ib1:
  # [orig l.118] how: `for item in list:` runs the indented lines once per entry, with `setting` equal to each pair in
  #                   turn.
  for setting in args.ib1:
    # [orig l.120] how: setting[0] is the first number of the pair (the node), setting[1] the second (the value).
    #                   int() makes the node a whole number (the flag stored 2.0) because list positions must be whole
    #                   numbers. `array[k] = v` overwrites entry k.
    # significance: the per-node exception: that node's input bias, uA -> A.
    transponder_bias1[int(setting[0])] = setting[1]*1.0e-6
# [orig l.121]
if args.ib2:
  # [orig l.122]
  for setting in args.ib2:
    # [orig l.123] significance: the per-node exception: that node's output bias, uA -> A.
    transponder_bias2[int(setting[0])] = setting[1] *1.0e-6

# [orig l.126] significance: the split fraction. Used in ONE place only: RHS step 3, orig l.210, to set the pulse size
#                            sent to the next gates.
transponder_current_divider = args.split

# [orig l.129] significance: U3's resistance R_nw1(t) for each node: 0 = superconducting. RHS updates it every step.
transponder_Rnw_input = np.zeros(N_transponders)
# [orig l.130] significance: loop 3's resistance R_T(t) for each node: R2, plus U2 while it is switched. RHS updates
#                            it every step.
transponder_RT_loop   = RT0*np.ones(N_transponders)
# [orig l.131] significance: a record of the gate current, for plotting only.
transponder_input_gate_current = np.zeros(N_transponders)
# [orig l.132] significance: a record of (U3 current + gate current), for plotting only.
transponder_source_current = np.zeros(N_transponders)

# [orig l.134] how: no copy is made: both names refer to the same array.
# significance: each U3 starts out carrying its whole bias current.
init_input_current  = transponder_bias1
# [orig l.135] significance: the storage loops start empty.
init_loop_currents  = np.zeros(N_transponders)


# [orig l.138] how: np.empty(n, dtype=...) reserves n slots without filling them (the next two lines fill them);
#                   dtype= makes them the same kind of number as transponder_RT_loop (decimal numbers).
# significance: the starting values given to the solver: 2 numbers per node = 16.
transponder_initial_state =  np.empty((2*N_transponders), dtype=transponder_RT_loop.dtype)
# [orig l.139] how: slice notation [start:stop:step]: [0::2] means every 2nd slot starting at slot 0 (0, 2, 4, ...
#                   14). The right side has 8 values for those 8 slots.
# significance: even slots = U3 channel current i_nw, starting at the bias.
transponder_initial_state[0::2] = init_input_current
# [orig l.140] how: [1::2] = every 2nd slot starting at slot 1 (1, 3, 5, ... 15).
# significance: odd slots = storage-loop current i_loop, starting at 0.
transponder_initial_state[1::2] = init_loop_currents


# ---------------- STEP 7: flip the wiring table to 'who sends to me?' ----------------
# [orig l.144] how: same list comprehension as above: N separate empty lists.
# significance: will hold, for each receiver, the nodes that send to it.
input_connections = [[] for _ in range(len(input_states))] 
# [orig l.145] how: `.T` transposes a table: rows become columns and columns become rows.
# significance: now row = receiver, column = sender.
input_delays = delay.T
# [orig l.146] how: enumerate(x) hands out pairs (position, item); `for i, arow in ...` unpacks each pair, so i = 0,
#                   1, 2... and arow = that row of the table.
# significance: each row of the flipped table = one receiver.
for i,arow in enumerate(connectivity.T):
  # [orig l.147] how: same idea one level down: j = position in the row, acol = the value there.
  # significance: each entry = one possible sender.
  for j,acol in enumerate(arow):
    # [orig l.148] how: `==` tests equality (a single `=` would assign).
    # significance: a link exists from j to i.
    if acol == 1:
      # [orig l.149] how: `list.append(x)` adds x at the end of the list.
      # significance: remember that node j sends to node i.
      input_connections[i].append(j)


# [orig l.154] significance: same as orig l.101, done a second time (harmless).
output_states = np.zeros(N_transponders)
# [orig l.156] significance: same as orig l.104, done a second time (harmless).
output_click_times = [[] for _ in range(len(input_states))]



# ---------------- STEP 8: RHS, the engine ----------------
# [orig l.159] how: RHS takes the time t and an array x of 16 numbers (the currents) and returns 16 numbers (how fast
#                   each is changing). You never call it yourself: solve_ivp calls it many times.
# significance: THE ENGINE. At every tiny time step it checks who fires and then reports the slopes of all currents.
def RHS(t, x):
  # [orig l.162] how: `x[::2]` = every 2nd entry starting at entry 0 (x[0], x[2], x[4], ...).
  # significance: the U3 channel current of every node.
  nw_currents   = x[::2]  
  # [orig l.163] how: `x[1::2]` = every 2nd entry starting at entry 1.
  # significance: the storage-loop current of every node.
  loop_currents = x[1::2]

  # [orig l.165] how: 1.0e-11 = 10^-11 s.
  # significance: 10 ps: 'close enough' when matching times.
  eps = 1.0e-11

    # ===== RHS step 1: the sensors fire (orig l.167-175). The outside world pokes the network =====
  # [orig l.167] how: enumerate gives (0, first list), (1, second list): n = which sensor, clicks = its list of
  #                   scheduled click times.
  for n, clicks in enumerate(sensor_pulses):
    # [orig l.168] how: a loop inside the loop: i_click = position, sensor_click_t = that scheduled time.
    for i_click, sensor_click_t in enumerate(clicks):  
      # [orig l.169] how: abs() is the absolute value. The solver only visits certain times t, so testing exact
      #                   equality would almost never be true; instead we ask whether t is within eps of the scheduled
      #                   click, on either side.
      # significance: it is now (within 10 ps) the time this sensor clicks.
      if( abs(t -sensor_click_t) <= eps ) :
        # [orig l.170] how: an f-string: anything inside {} is replaced by its value.
        # significance: debug output.
        print(f"n = {n} i_click = {i_click} time {t} vs {sensor_click_t}")
        # [orig l.171] significance: the sensor is marked as clicked.
        output_states[n] = 1
        # [orig l.174] significance: RECORD THE CLICK TIME. Every node listening to this sensor will find it in RHS
        #                            step 3.
        output_click_times[n].append(t)
        # [orig l.175] how: `list.pop(k)` removes the k-th entry.
        # significance: so each sensor fires only once.
        sensor_pulses[n].pop(i_click)


    # ===== RHS step 2: output nTron U2 (orig l.177-197). Did each transponder fire? =====
  # [orig l.178] how: i_out = node number, output_state = its state value (not used again; the code re-reads
  #                   output_states[i_out]).
  for i_out, output_state in enumerate(output_states):
    # [orig l.179] significance: U2 is switched (it fired recently).
    if output_states[i_out] == 1:
      # [orig l.183] how: list[len(list)-1] is the last entry (positions start at 0, so the last position is len-1).
      #                   output_click_times[i_out] is node i_out's list of spike times.
      # significance: when it last fired.
      tlast                       = output_click_times[i_out][len(output_click_times[i_out])-1]
      # [orig l.185] how: LinearPulse(t, t0=tlast, width=2.0e-9) returns a number 0..1 (shape: see STEP 2); Rnorm
      #                   times that is ohms (0..1000); adding RT0 gives the loop's total resistance.
      # significance: THIS IS WHERE LinearPulse BECOMES A RESISTANCE FOR U2: while switched, loop 3's resistance is R2
      #               plus up to 1000 ohm. That dumps the stored loop current (it becomes RT in the y1/y2 equations).
      transponder_RT_loop[i_out]  = RT0 + Rnorm*LinearPulse(t,t0=tlast,width=2.0e-9)
      # [orig l.187] significance: 3 ns have passed since it fired.
      if t-tlast > ref_period:
        # [orig l.188] significance: U2 is ready to fire again.
        output_states[i_out] = 0
        # [orig l.189] significance: debug output.
        print("resetting output ntron")
    # [orig l.190] significance: U2 is ready.
    if output_states[i_out] == 0:
      # [orig l.191] significance: loop 3's resistance is just R2.
      transponder_RT_loop[i_out] = RT0
      # [orig l.193] how: adds node i_out's storage-loop current and its output bias, and compares the sum with the
      #                   threshold.
      # significance: THE FIRING TEST: storage-loop current (into U2's gate) + bias4 beats U2's threshold.
      if loop_currents[i_out] + transponder_bias2[i_out] > Inw2_sw: 
        # [orig l.195] how: `{}` is filled by the values given to .format(...).
        # significance: debug output.
        print("output {}: click_t = {}".format(i_out, t))
        # [orig l.196] significance: U2 switches.
        output_states[i_out] = 1
        # [orig l.197] significance: RECORD THE SPIKE TIME. This is the moment current leaves the transponder.
        output_click_times[i_out].append(t)


    # ===== RHS step 3: input nTron U3 (orig l.199-236). Did enough arrive from upstream? =====
  # [orig l.201]
  for i_in, input_state in enumerate(input_states):
    # [orig l.202] significance: U3 is ready.
    if input_states[i_in] == 0:
      # [orig l.203] significance: superconducting: zero resistance.
      transponder_Rnw_input[i_in] = 0
      # [orig l.205] significance: start adding up the gate current from zero.
      ntron_gate_current = 0
      # [orig l.206] significance: plotting record.
      transponder_input_gate_current[i_in] = 0
      # [orig l.207] how: input_connections[i_in] is the list of senders to node i_in; the loop visits each sender
      #                   j_in.
      for j_in in input_connections[i_in]:
        # [orig l.208] how: connectivity[j_in] is row j_in of the table; np.sum adds its entries, i.e. counts the
        #                   links leaving j_in.
        # significance: the sender's fan-out.
        n_split = np.sum(connectivity[j_in])
        # [orig l.209] significance: the sender's output bias (bias4).
        b2  = transponder_bias2[j_in]
        # [orig l.210] how: plain arithmetic: sender's bias x split fraction / fan-out.
        # significance: THE PULSE SIZE from this sender. The only place `split` is used.
        amp = b2*transponder_current_divider/n_split
        # [orig l.212] significance: every spike this sender has ever fired (older ones have decayed to ~0).
        for click_t in output_click_times[j_in]:
          # [orig l.213] how: input_delays[i_in][j_in] = row i_in, column j_in of the flipped table.
          # significance: the link delay (the delay box in the schematic).
          click_delay         = input_delays[i_in][j_in]
          # [orig l.214] significance: when that spike arrives at node i_in.
          click_arrival       = click_t  + click_delay
          # [orig l.215] how: `a += b` means a = a + b. OutputPulse(t, t0=click_arrival) is the pulse shape (0..0.61)
          #                   evaluated at the current time.
          # significance: this pulse's current right now is added to the gate current (loop 1). Pulses from several
          #               senders add up.
          ntron_gate_current += amp*OutputPulse(t, t0=click_arrival)
      # [orig l.218] significance: plotting record.
      transponder_source_current[i_in]     = nw_currents[i_in] + ntron_gate_current
      # [orig l.219] significance: plotting record.
      transponder_input_gate_current[i_in] = ntron_gate_current
      # [orig l.220] significance: THE CLICK TEST: U3 channel current + gate current beats U3's threshold.
      if nw_currents[i_in] + ntron_gate_current > Inw1_sw: 
        # [orig l.222] significance: U3 clicks.
        input_states[i_in] = 1
        # [orig l.223] significance: record the click time.
        input_click_times[i_in].append(t)
    # [orig l.226] significance: U3 is clicked (resistive).
    if input_states[i_in] == 1:
      # [orig l.228] how: same 'last entry of a list' idiom as orig l.183.
      # significance: when it clicked.
      latest_input_click = input_click_times[i_in][len(input_click_times[i_in])-1]
      # [orig l.229] significance: 3 ns have passed since the click.
      if t-latest_input_click > ref_period:
        # [orig l.231] significance: debug output.
        print("input reset")
        # [orig l.232] significance: U3 is ready again.
        input_states[i_in] = 0
      # [orig l.234] significance: less than 3 ns since the click.
      else: 
        # [orig l.236] how: LinearPulse(t, t0=latest_input_click, width=1.0e-9) returns 0..1 (STEP 2); times Rnorm =
        #                   ohms.
        # significance: THIS IS WHERE LinearPulse BECOMES A RESISTANCE FOR U3 (Rnw): 0 -> 1000 ohm over 1 ns, 1000 ohm
        #               for 1 ns, back to 0 over 0.5 ns (2.5 ns total, then 0 until ref_period ends). It pushes U3's
        #               current into the shunt and storage loop.
        transponder_Rnw_input[i_in] = Rnorm*LinearPulse(t,t0=latest_input_click, width=1.0e-9)

    # ===== RHS step 4: circuit physics (orig l.237-250). Where does the current go? =====
  # [orig l.237] how: [] is an empty list.
  # significance: will hold the slope of every current.
  y = []
  # [orig l.238]
  for i_node,loop_cur in enumerate(loop_currents):
    # [orig l.239] significance: bias3 flowing into this node.
    i1  = transponder_bias1[i_node]
    # [orig l.240] significance: current in U3 right now.
    inw = nw_currents[i_node]
    # [orig l.241] significance: current in the storage loop right now.
    iloop = loop_cur
    # [orig l.242] significance: U3's resistance right now (from RHS step 3).
    Rnw =  transponder_Rnw_input[i_node]
    # [orig l.243] significance: loop 3's resistance right now (from RHS step 2).
    RT  = transponder_RT_loop[i_node]
    # [orig l.246] how: ordinary arithmetic: `/` divides the whole bracketed numerator by the whole bracketed
    #                   denominator.
    # significance: y1 = d(i_nw)/dt, how fast U3's current is changing (Kirchhoff's laws with the inductors). Caution:
    #               the Lnw*Ls1 term has the opposite sign from the circuit as drawn (about 5% on time constants);
    #               worth checking with Whit.
    y1 = (-inw*Ls1*Rnw - i1*LT*Rs1 + iloop*LT*Rs1 + inw*LT*(Rnw + Rs1) + iloop*Ls1*RT)/(Lnw*(Ls1 - LT) - Ls1*LT)
    # [orig l.247] significance: y2 = d(i_loop)/dt, how fast the storage-loop current is changing.
    y2 = (-inw*Ls1*Rnw - i1*Lnw*Rs1 + iloop*Lnw*Rs1 + inw*Lnw*Rs1 + iloop*Lnw*RT + iloop*Ls1*RT)/(Lnw*(Ls1 - LT) - Ls1*LT)
    # [orig l.248]
    y.append(y1)
    # [orig l.249] significance: y ends up as [slope of i_nw(0), slope of i_loop(0), slope of i_nw(1), ...], the same
    #                            order as x.
    y.append(y2)
  # [orig l.250] significance: hands all 16 slopes back to the solver, which takes the next step and calls RHS again.
  return y




# ---------------- STEP 9: main(), run the simulation and make the original plot ----------------
# [orig l.254] how: `-> int` just declares that main returns a whole number.
def main() -> int:
  """ Prototype network simulation"""
  # [orig l.256]
  print(delay)
  


  # [orig l.271] how: np.linspace(a, b, num=n) gives n evenly spaced numbers from a to b inclusive: here 1000 times
  #                   from 0 to 100 ns.
  # significance: the times at which results are saved.
  t_pts = np.linspace(0, 100*1.0e-9, num=1000)

  # [orig l.273]
  print(f"shunt/output current splilt: {transponder_current_divider}")

  # [orig l.275]
  print("Starting simulation")
  # [orig l.276] how: solve_ivp(fun, (t_start, t_end), start_values, t_eval=..., max_step=...) solves dx/dt = fun(t,
  #                   x): from start_values at t_start it repeatedly calls fun to get the slopes and advances x by a
  #                   small step (by default an adaptive Runge-Kutta method). t_eval = the times at which to store
  #                   results; max_step caps the step size. The result has .t (times) and .y (a 16 x 1000 table: one
  #                   row per current, one column per saved time).
  # significance: RUNS THE SIMULATION (about a minute). The 1 ps step cap stops it skipping over sub-nanosecond
  #               switching events.
  solution = solve_ivp(RHS, (0, 100 * 1.0e-9), transponder_initial_state, t_eval= t_pts, max_step = 1.0e-12)

  # [orig l.278] how: plt.subplots(6) creates a figure with 6 stacked panels and returns the figure and an array of
  #                   the 6 panels.
  # significance: the original plot: for t1, t2, t3 the U3 current (labelled 'gate') and the loop current (red).
  fig, axs = plt.subplots(6)
  # [orig l.280] how: `.plot(x, y, label=...)` draws a line. solution.y[4] = row 4 of the result table = i_nw of node
  #                   2 (t1) (row 2n is i_nw of node n). Dividing by 1e-6 gives microamps and solution.t/1e-9 gives
  #                   ns.
  axs[0].plot(solution.t/1.0e-9, solution.y[4]/1.0e-6, label='gate 2')
  # [orig l.281] how: row 5 = i_loop of node 2; 'r' = red.
  axs[1].plot(solution.t/1.0e-9, solution.y[5]/1.0e-6, 'r',label='loop 2')
  # [orig l.282] how: row 6 = i_nw of node 3 (t2).
  axs[2].plot(solution.t/1.0e-9, solution.y[6]/1.0e-6, label='gate 3')
  # [orig l.283] how: row 7 = i_loop of node 3 (t2).
  axs[3].plot(solution.t/1.0e-9, solution.y[7]/1.0e-6, 'r',label='loop 3')
  # [orig l.284] how: row 8 = i_nw of node 4 (t3).
  axs[4].plot(solution.t/1.0e-9, solution.y[8]/1.0e-6, label='gate 4')
  # [orig l.285] how: row 9 = i_loop of node 4 (t3).
  axs[5].plot(solution.t/1.0e-9, solution.y[9]/1.0e-6, 'r',label='loop 4')

  # [orig l.291] how: writes text on the panel; the `\n` characters start new lines.
  # significance: the settings are printed on the figure.
  axs[0].annotate(f"dt={dt//1.0e-9}, \nb1={transponder_bias1/1.0e-6}, \nb2={transponder_bias2/1.0e-6}", xycoords="subfigure points",xy=(1,1))

  # [orig l.293]
  axs[0].legend()
  # [orig l.294]
  axs[1].legend()
  # [orig l.295]
  axs[2].legend()
  # [orig l.296]
  axs[3].legend()
  # [orig l.297]
  axs[4].legend()
  # [orig l.298]
  axs[5].legend()

  # [orig l.300] how: axs.flat lets a loop visit every panel.
  for ax in axs.flat:
    # [orig l.301]
    ax.set(xlabel='time [ns]',ylabel='current [uA]')

  # [orig l.304]
  for ax in axs.flat:
    # [orig l.305] how: hides axis labels on inner panels so they are not repeated.
    ax.label_outer()
  # [orig l.307] how: writes the figure to a file; {args.output} inserts the --output flag's text.
  plt.savefig(f"../results/{args.output}.png", bbox_inches='tight',dpi=200)
  # [orig l.308]
  plt.savefig(f"../results/{args.output}.pdf", format="pdf", bbox_inches='tight',dpi=200)

  # [orig l.311] significance: 0 means the solver finished without error.
  print(f"solution status {solution.status}")


    # ---------------- STEP 10 [new]: rebuild the gate current, then the three plots you asked for ----------------
  # [new] significance: the solver's saved time points, in seconds.
  t   = solution.t
  # [new] how: dividing the whole array by 1e-9.
  # significance: the same times in ns.
  tns = t/1.0e-9
  # [new] how: `list + list` joins two lists. [f't{k}' for k in range(1, N-1)] is a list comprehension making 't1' ...
  #            't6' (range(1, 7) = 1..6).
  # significance: labels for the 8 nodes.
  names = ['i1', 'i2'] + [f't{k}' for k in range(1, N_transponders-1)]
  # [new] how: a dictionary stores key -> value pairs; `dict()` is an empty one. One per node.
  # significance: gate_parts[i][j] will hold the current node i receives from node j. RHS computed these but threw
  #               them away.
  gate_parts = [dict() for _ in range(N_transponders)]
  # [new] how: np.zeros((rows, columns)) makes a table of zeros; the double brackets pass the shape as one pair.
  # significance: the total gate current of each node over time.
  gate_current = np.zeros((N_transponders, len(t)))
  # [new] how: range(N) is 0, 1, ..., N-1.
  # significance: each receiver.
  for i_in in range(N_transponders):
    # [new] significance: each node that feeds it.
    for j_in in input_connections[i_in]:
      # [new] significance: pulse size = sender's Ib2 x split / fan-out (same as orig l.208-210).
      amp  = transponder_bias2[j_in]*transponder_current_divider/np.sum(connectivity[j_in])
      # [new] how: len(t) is the number of saved times.
      # significance: this sender's contribution over time, starting at 0.
      part = np.zeros(len(t))
      # [new]
      for click_t in output_click_times[j_in]:
        # [new] significance: when that spike reaches i_in (orig l.213-214).
        arrival = click_t + input_delays[i_in][j_in]
        # [new] how: a list comprehension over every saved time tk: `A if condition else B` gives A when the condition
        #            is true and B otherwise; np.array turns the list into an array so it can be added to `part` entry
        #            by entry.
        # significance: the pulse only counts after the spike happened, as RHS cannot see the future.
        part += np.array([amp*OutputPulse(tk, t0=arrival) if tk >= click_t else 0.0 for tk in t])
      # [new] how: stores part in the dictionary under the key j_in.
      gate_parts[i_in][j_in] = part
      # [new] significance: adds this sender's piece to node i_in's total.
      gate_current[i_in] += part

  # [new]
  print("output (spike) click times [ns]:")
  # [new]
  for n in range(N_transponders):
    # [new] how: np.round(array, 2) rounds every entry to 2 decimals.
    # significance: spike times in ns.
    print(f"  {names[n]}: {np.round(np.array(output_click_times[n])/1.0e-9, 2)}")


    # ---- SPIKE RASTER ----
  # [new] how: plt.subplots() with no number makes one panel; figsize is (width, height) in inches.
  # significance: SPIKE RASTER: one row per node, a black tick for each spike, a red triangle for each input click.
  fig, ax = plt.subplots(figsize=(8, 3.5))
  # [new]
  for n in range(N_transponders):
    # [new] how: `ax.vlines(x_values, ymin, ymax, ...)` draws one vertical line at each x from ymin to ymax.
    #            np.array(list)/1e-9 converts the spike times to ns. `label=... if n == 0 else None` gives a legend
    #            label only to the first node so it appears once.
    # significance: one black tick per spike.
    ax.vlines(np.array(output_click_times[n])/1.0e-9, n-0.4, n+0.4, color='k', lw=2,
              label='output click (spike)' if n == 0 else None)
    # [new] how: an empty list counts as false.
    if input_click_times[n]:
      # [new] how: `[n-0.45]*k` is a list with k copies of n-0.45 (the height of each marker); 'v' is a downward
      #            triangle marker; ms = marker size. `any(list)` is true if any entry is true, so the legend label is
      #            attached only to the first node that has clicks.
      # significance: one red triangle per input click.
      ax.plot(np.array(input_click_times[n])/1.0e-9, [n-0.45]*len(input_click_times[n]), 'v', color='tab:red', ms=4,
              label='input (gate) click' if not any(input_click_times[:n]) else None)
  # [new] how: a loop over the two values in the bracket.
  for y in (1.5, 4.5):
    # [new] how: draws a horizontal line across the plot at height y.
    # significance: separates sensors | t1-t3 | t4-t6.
    ax.axhline(y, color='0.8', ls='--', lw=0.8)
  # [new] how: puts a tick at each row number and labels it with the matching name.
  ax.set_yticks(range(N_transponders), names)
  # [new] how: the range is given top value first, which flips the axis so row 0 is at the top.
  # significance: i1 at the top.
  ax.set_ylim(N_transponders-0.5, -0.5)
  # [new] how: tns[-1] is the last entry of tns (negative positions count from the end).
  ax.set(xlim=(0, tns[-1]), xlabel='time [ns]', title='spike raster')
  # [new]
  ax.legend(loc='upper right', fontsize=8, frameon=False)
  # [new]
  plt.savefig(f"../results/{args.output}_raster.png", bbox_inches='tight', dpi=200)
  # [new]
  plt.savefig(f"../results/{args.output}_raster.pdf", bbox_inches='tight')


    # ---- GATE-CURRENT TRACE ----
  # [new] significance: GATE-CURRENT TRACE for one node: do the incoming pulses add up above threshold?
  n = args.trace_node
  # [new] how: row 2n of the result table is node n's U3 current.
  # significance: node n's U3 channel current over time.
  inw = solution.y[2*n]
  # [new] how: two stacked panels; sharex=True gives them the same time axis.
  fig, axs2 = plt.subplots(2, sharex=True, figsize=(8, 5))
  # [new] how: `.items()` gives a dictionary's (key, value) pairs.
  # significance: each sender and its pulse.
  for j_in, part in gate_parts[n].items():
    # [new] how: '--' draws a dashed line.
    # significance: TOP panel: each sender's pulse.
    axs2[0].plot(tns, part/1.0e-6, '--', label=f'from {names[j_in]}')
  # [new] how: 'k' = black; lw = line width.
  # significance: the sum of the pulses.
  axs2[0].plot(tns, gate_current[n]/1.0e-6, 'k', lw=1.5, label='summed gate current')
  # [new] how: ls=':' is a dotted line.
  # significance: the gate current needed to make U3 click (threshold minus bias).
  axs2[0].axhline((Inw1_sw - transponder_bias1[n])/1.0e-6, color='r', ls=':', label='needed to switch (I1sw - Ib1)')
  # [new]
  axs2[0].set(ylabel='gate current [uA]', title=f'{names[n]}: gate current')
  # [new]
  axs2[0].legend(fontsize=8, frameon=False)
  # [new] significance: BOTTOM panel: U3's own current.
  axs2[1].plot(tns, inw/1.0e-6, color='0.5', label='channel current i_nw')
  # [new] significance: what the click test compares with the threshold (orig l.220).
  axs2[1].plot(tns, (inw + gate_current[n])/1.0e-6, 'k', label='i_nw + gate')
  # [new] significance: U3's threshold.
  axs2[1].axhline(Inw1_sw/1.0e-6, color='r', ls=':', label='I1sw (threshold)')
  # [new]
  for k, tc in enumerate(input_click_times[n]):
    # [new] how: draws a vertical line; alpha=0.4 makes it faint.
    # significance: marks each click.
    axs2[1].axvline(tc/1.0e-9, color='r', alpha=0.4, label='input click' if k == 0 else None)
  # [new]
  axs2[1].set(xlabel='time [ns]', ylabel='current [uA]')
  # [new]
  axs2[1].legend(fontsize=8, frameon=False)
  # [new]
  plt.savefig(f"../results/{args.output}_gate_trace_{names[n]}.png", bbox_inches='tight', dpi=200)
  # [new]
  plt.savefig(f"../results/{args.output}_gate_trace_{names[n]}.pdf", bbox_inches='tight')


    # ---- GRADIENT RASTER ----
  # [new] how: `A if condition else B`: the largest value in gate_current if it is above 0, otherwise 1 (so the next
  #            line never divides by 0).
  # significance: GRADIENT RASTER: the largest gate current anywhere in the run is pitch black.
  gmax  = gate_current.max() if gate_current.max() > 0 else 1.0
  # [new] how: dividing a whole table by a number divides every entry.
  # significance: every value now lies between 0 (no current) and 1 (maximum).
  shade = gate_current/gmax
  # [new]
  for n in range(N_transponders):
    # [new]
    for tc in input_click_times[n]:
      # [new] how: (t > tc) is an array of True/False, one per saved time; `&` is an element-wise AND of two such
      #            arrays; using that array as the index selects only the times where it is True, and `= 0` sets just
      #            those entries to 0.
      # significance: white for ref_period right after a click: the nanowire is recovering.
      shade[n][(t > tc) & (t <= tc + ref_period)] = 0
  # [new]
  fig, ax = plt.subplots(figsize=(8, 3.5))
  # [new] how: imshow draws a table of numbers as coloured cells. cmap='Greys' maps 0 to white and 1 to black
  #            (vmin/vmax set that scale); extent=(left, right, bottom, top) says which x and y range the picture
  #            covers; aspect='auto' stretches it to fit.
  # significance: the gradient itself: one row per node, time along x.
  im = ax.imshow(shade, aspect='auto', cmap='Greys', vmin=0, vmax=1, interpolation='nearest',
                 extent=(tns[0], tns[-1], N_transponders-0.5, -0.5))
  # [new]
  for n in range(N_transponders):
    # [new] significance: solid red = spike.
    ax.vlines(np.array(output_click_times[n])/1.0e-9, n-0.5, n+0.5, color='red', lw=1.8,
              label='output click (spike)' if n == 0 else None)
    # [new] significance: dotted red = input click.
    ax.vlines(np.array(input_click_times[n])/1.0e-9, n-0.5, n+0.5, color='red', lw=0.8, ls=':',
              label='input (gate) click' if n == 2 else None)
  # [new] how: bbox_to_anchor=(0.5, -0.15) positions the legend below the plot.
  ax.legend(loc='upper center', fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.15))
  # [new]
  for y in (1.5, 4.5):
    # [new] significance: separators.
    ax.axhline(y, color='0.8', ls='--', lw=0.8)
  # [new]
  ax.set_yticks(range(N_transponders), names)
  # [new]
  ax.set(xlabel='time [ns]', title='gradient raster: gate current (white = 0, black = max)')
  # [new] how: adds a colour scale; {gmax/1.0e-6:.2f} prints gmax in microamps with 2 decimals.
  # significance: reads the shading back as a fraction of the largest gate current.
  fig.colorbar(im, ax=ax, label=f'gate current / {gmax/1.0e-6:.2f} uA', pad=0.01)
  # [new]
  plt.savefig(f"../results/{args.output}_gradient_raster.png", bbox_inches='tight', dpi=200)
  # [new]
  plt.savefig(f"../results/{args.output}_gradient_raster.pdf", bbox_inches='tight')

  # [orig l.314] significance: main finished successfully.
  return 0

# [orig l.316] how: `__name__` equals '__main__' when the file is run directly (python toy_example.py) and equals the
#                   file's name when it is imported from another file, so main() only runs in the first case.
if __name__ == '__main__':
  # [orig l.317] how: main() runs and returns 0; sys.exit(0) ends the program and reports success.
  sys.exit(main())
