# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # N06 — toy_example walkthrough
#
# `transponder_model-whit_0918/src/tests/toy_example.py`, cut into small steps so you can see what
# each part produces. The code is the same as the script (same pulse shapes, same `RHS`, same
# post-processing); the only structural changes are:
#
# * the command-line flags are plain variables in the **Parameters** cell — edit them there;
# * the script's global state is rebuilt by `reset_state()`, which you must run before every solve
#   (see section 5 for why);
# * section 4 adds a single-transponder run that is **not** in the script, to separate "what one
#   node does" from "what the wiring does".
#
# Runtime: about a minute per network solve; the whole notebook takes ~6 minutes.

# %%
# [orig l.5] maths functions (OutputPulse uses exp)
import math
# [orig l.1] arrays (the network tables and current vectors)
import numpy as np
# [orig l.3] plotting
import matplotlib.pyplot as plt
# [orig l.2] the differential-equation solver that steps the circuit forward in time
from scipy.integrate import solve_ivp

# [new] notebook only: figure resolution
plt.rcParams["figure.dpi"] = 110

# %% [markdown]
# ## 1. Parameters
#
# These replace the script's command-line flags. The values below are the settings used in this
# walkthrough (chosen to reproduce the traces you showed from Whit); the script's own defaults are
# `--Ib1init 16 --Ib2init 16 --I1sw 20 --I2sw 20 --split 0.1 --dt 5`. Equivalent command:
#
# ```
# python toy_example.py --dt 35 --Ib1init 12 --ib1 2 13 --ib1 4 11.2 --Ib2init 19 --I1sw 14 --split 0.5
# ```
#
# **Comment key used on every variable below**
# * `meaning`   — what the quantity is physically
# * `schematic` — which part of `single_transponder_simple.pdf` it is
# * `vs original` — how it differs from `toy_example.py` on `main` (the script as Whit wrote it)
# * `does`      — what the line does in the code
# * `orig l.N` / `[orig l.N]` — line number in `toy_example.py` on `main` (Whit's original, 317 lines);
#   `[new]` marks code that does not exist in the original
#
# Biases are in **µA** here and converted to amps below, as in the script.
# Per-node overrides use the **matrix row index**: 0 = i1, 1 = i2, 2 = t1, … 7 = t6.

# %%
# ---- the knobs ------------------------------------------------------------------------------
dt_ns = 35.0
# meaning:     time between the two sensor clicks [ns]
# schematic:   pulser2 (U4) fires this long after pulser1 (U1)
# vs original: was flag --dt (args.dt, default 5 ns); now a variable named dt_ns, set to 35
# orig line:   flag --dt at l.49; used at l.106 (dt = 1.0e-9*args.dt) and l.112 (sensor_pulses)
# does:        converted to seconds (dt) below; used in sensor_pulses

Ib1init = 12.0
# meaning:     input bias current given to every node [uA]
# schematic:   I_b1, "bias3 in" -> U3 drain (LTspice uses ~150 uA)
# vs original: was flag --Ib1init (default 16); now a variable, set to 12
# orig line:   flag --Ib1init at l.43; converted at l.61
# does:        becomes Ib1, the starting value of i_nw and the bias in the circuit equations

Ib2init = 19.0
# meaning:     output bias current given to every node [uA]
# schematic:   I_b2, "bias4 in" -> U2 drain (LTspice uses ~160 uA)
# vs original: was flag --Ib2init (default 16); now a variable, set to 19
# orig line:   flag --Ib2init at l.44; converted at l.62
# does:        becomes Ib2; sets the firing test (orig l.193) and the pulse size sent downstream (orig l.210)

I1sw_uA = 14.0
# meaning:     switching current (threshold) of the input nTron [uA]
# schematic:   U3 (input nTron, Q1)
# vs original: was flag --I1sw (args.I1sw, default 20); now a variable named I1sw_uA, set to 14
# orig line:   flag --I1sw at l.45; converted at l.58
# does:        becomes Inw1_sw; the input-click test is i_nw + gate > Inw1_sw (orig l.220)

I2sw_uA = 20.0
# meaning:     switching current (threshold) of the output nTron [uA]
# schematic:   U2 (output nTron, Q2)
# vs original: was flag --I2sw (args.I2sw, default 20); renamed I2sw_uA, same value
# orig line:   flag --I2sw at l.46; converted at l.59
# does:        becomes Inw2_sw; the firing test is i_loop + Ib2 > Inw2_sw (orig l.193)

split = 0.5
# meaning:     fraction of a spike's output current sent on to the nodes it feeds (rest goes to a shunt)
# schematic:   no single part; stands in for the L2 / R1 current division of loop 4 (R1 = R_s2)
# vs original: was flag --split (args.split, default 0.1); now a variable, set to 0.5
# orig line:   flag --split at l.47; assigned at l.126
# does:        copied to transponder_current_divider; used only in amp = Ib2*split/n (orig l.210)

ib1_overrides = {2: 13.0, 4: 11.2}
# meaning:     per-node exceptions to Ib1init: {node number: bias in uA}; 2 = t1, 4 = t3
# schematic:   bias3 of one particular box (t1, t3)
# vs original: was flag --ib1 NODE VALUE, repeatable (args.ib1, a list of pairs); now a dict. The
#              dict values (13, 11.2) are mine; the script's default is no overrides
# orig line:   flag --ib1 at l.51; applied at l.117-120
# does:        reset_state() overwrites transponder_bias1[node] with each value (orig l.117-120 does this from args.ib1)

ib2_overrides = {}
# meaning:     per-node exceptions to Ib2init
# schematic:   bias4 of one particular box
# vs original: was flag --ib2 NODE VALUE (args.ib2); now a dict, empty as in the script's default
# orig line:   flag --ib2 at l.52; applied at l.121-123
# does:        reset_state() overwrites transponder_bias2[node] (orig l.121-123 does this from args.ib2)

T_END = 100e-9
# meaning:     simulated time [s]
# schematic:   none
# vs original: NEW variable; the script writes 100*1e-9 directly in t_pts and solve_ivp
# orig line:   `100*1.0e-9` written at l.271 (t_pts) and l.276 (solve_ivp)
# does:        end of the time axis in solve_network() and the plots

# %%
# ---- unit conversion and fixed circuit values ----------------------------------------------
# [orig l.58] U3 threshold: microamps -> amps
Inw1_sw = I1sw_uA * 1.0e-6     # input ntron switching current [A]
# [orig l.59] U2 threshold: microamps -> amps
Inw2_sw = I2sw_uA * 1.0e-6     # output ntron switching current [A]
# [orig l.61] input bias, microamps -> amps
Ib1 = Ib1init * 1.0e-6         # schematic: I_b1 (bias3)
# [orig l.62] output bias, microamps -> amps
Ib2 = Ib2init * 1.0e-6         # schematic: I_b2 (bias4)
# [orig l.106] sensor spacing, ns -> s
dt  = dt_ns * 1.0e-9
# [orig l.126] same number, original name (used in RHS step 3)
transponder_current_divider = split
# vs original: same conversions. Only the source changes: args.X -> the variables above.
#              The script's --generator option (time-of-flight generator sets dt) is not included.

# orig line:   l.65
Ls1 = 5 * 1.0e-9
# meaning:     inductance of the shunt branch around U3
# schematic:   L1 (5 nH), "L_s1", loop 2
# vs original: unchanged
# does:        in y1/y2 (orig l.246-247)

# orig line:   l.66
Rs1 = 2
# meaning:     resistance of the shunt branch around U3 [ohm]
# schematic:   R3, "R_s1", loop 2 (R3 = 3 ohm in LTspice, 2 ohm here)
# vs original: unchanged
# does:        in y1/y2

# orig line:   l.67
LT = 150 * 1.0e-9
# meaning:     inductance of the storage loop
# schematic:   Lb1 (~148 nH), "L_T", loop 3
# vs original: unchanged
# does:        in y1/y2

# orig line:   l.68
RT0 = 3
# meaning:     resistance of the storage loop when U2 is not switched [ohm]
# schematic:   R2, "R_T", loop 3 (R2 = 4 ohm in LTspice, 3 ohm here)
# vs original: unchanged
# does:        RT = RT0 normally (orig l.191), RT0 + a 1000 ohm pulse while U2 fires (orig l.185)

# orig line:   l.70
Lnw = 10 * 1.0e-9
# meaning:     inductance of an nTron channel
# schematic:   L_nw1 inside U3 (no separate part)
# vs original: unchanged
# does:        in y1/y2

# orig line:   l.71
Rnorm = 1000
# meaning:     resistance of an nTron channel when it is switched [ohm]
# schematic:   R_nw1(t) of U3; for U2 it is added onto R2
# vs original: unchanged
# does:        scales LinearPulse into a resistance (orig l.185 for U2, l.236 for U3)

# orig line:   l.75
ref_period = 0.003 * 1.0e-6
# meaning:     rest time (3 ns) after a click before the same nTron can click again
# schematic:   none (model bookkeeping)
# vs original: unchanged value. The script's comment says "also not used??"; it is used (orig l.187 and l.229)
# does:        refractory timer in RHS and the white gap in the gradient raster

print(f"input threshold {Inw1_sw*1e6} uA, output threshold {Inw2_sw*1e6} uA, split {split}")

# %% [markdown]
# ### Section 1 variables ↔ schematic (bluntly)
#
# | variable | schematic part | what it does |
# |---|---|---|
# | `Ib1init`, `Ib1`, `ib1_overrides` | **bias3 / I_b1** → U3 drain | current through U3's channel at rest. U3 clicks when this + gate current > `Inw1_sw` |
# | `Ib2init`, `Ib2`, `ib2_overrides` | **bias4 / I_b2** → U2 drain | sets the size of the pulse a spike sends out: `Ib2·split/n` |
# | `I1sw_uA`, `Inw1_sw` | **U3** switching current | threshold of the input nTron |
# | `I2sw_uA`, `Inw2_sw` | **U2** switching current | threshold of the output nTron (`i_loop + Ib2 > Inw2_sw` → spike) |
# | `split`, `transponder_current_divider` | **output junction: L2 / R1** | **only** scales the gate pulse delivered to downstream U3 gates (loop ①). Does not touch `i_nw`, `i_loop`, the shunt, or the receiver's bias |
# | `Ls1`, `Rs1` | **L1, R3** (loop ②) | shunt branch around U3 |
# | `LT`, `RT0` | **Lb1, R2** (loop ③) | storage loop |
# | `Lnw`, `Rnorm` | inside **U3 / U2** | channel inductance / normal-state resistance |
# | `ref_period` | none | bookkeeping: lockout after a click |
# | `dt_ns`, `T_END` | none (pulsers i1/i2 timing, sim length) | |
# | `delay` matrix | **delay boxes** on each link | |
#
# Full picture of all six transponders wired as in this circuit: `docs/figures/N06_network_schematic.png`
# (regenerate with `python docs/figures/make_network_schematic.py --render`).
#
# ![six transponders](../docs/figures/N06_network_schematic.png)

# %% [markdown]
# ## 2. The two pulse shapes
#
# Everything in the model is built from two shapes:
#
# * **`LinearPulse`** — a trapezoid (rise, flat top, fall), scaled by `Rnorm`. It is the resistance
#   an nTron channel shows while it is switched to the normal state.
# * **`OutputPulse`** — a sigmoid rise times an exponential decay. It is the shape of the current
#   one spike delivers to a downstream gate; it peaks at about **0.61**, so a spike of nominal
#   amplitude `amp` delivers at most `0.61 * amp`.

# %% [markdown]
# **Section 2 ↔ schematic.** `LinearPulse(t0, width, rise, fall)` = R_nw(t) of U3 or U2 while it is
# resistive (`t0` = click time, `Rnorm` × 0..1). `OutputPulse(t0, tau, mu0)` = the current an upstream
# spike pushes through the delay line + R_in/L_in into loop ① (`t0` = click time + link delay;
# `tau`, `mu0` = rise/decay shape). Neither is a single physical part.

# %%
# vs original: LinearPulse same maths (the script stores t-t0 in a temp variable delta_t);
#              OutputPulse identical.
# [orig l.14] LinearPulse: a trapezoid between 0 and 1. Used as nTron resistance (x Rnorm) while it is switched.
#   t = now, t0 = when it starts, width = flat top, rise / fall = ramp times
def LinearPulse(t, t0=5.0e-9, width=0.5e-9, rise=1.0e-9, fall=0.5e-9):  # trapezoid, 0..1
  # [orig l.15-16] before the pulse starts or after it ends: 0 (nTron superconducting)
  if (t-t0 < 0) or (t-t0 > width+fall+rise):
    return 0
  # [orig l.17-21] ramping up: straight line from 0 to 1 over `rise`
  if ((t-t0) > 0) and ((t - t0) < rise):
    return (t-t0)/rise
  # [orig l.22-26] ramping down: straight line from 1 to 0 over `fall`
  if (t-t0 > width+rise) and (t-t0 <= width+fall+rise):
    return ((t0+width+fall+rise)-t)/fall
  # [orig l.27-28] otherwise we are on the flat top: 1
  else:
    return 1

# [orig l.30] OutputPulse: the current one upstream spike delivers into this node's gate (loop 1).
#   t0 is where the pulse is about half way up; tau = decay time; mu0 = how sharp the rise is
def OutputPulse(t, t0=1e-9, tau=1.0e-9, mu0=0.2e-9):  # sigmoid rise at t0, exp decay with tau
  # [orig l.33] (smooth step up around t0) x (exponential decay). It peaks at about 0.61 and is already ~20% of nominal 0.35 ns BEFORE t0
  return (1.0 - 1.0/(math.exp((t - t0)/mu0) + 1.0))*math.exp(- ((t - t0)/tau))

# [new] --- STEP: draw both shapes so you can see them ---
# time axis 0-8 ns, 800 points
ts = np.linspace(0, 8e-9, 800)
fig, axs = plt.subplots(1, 2, figsize=(9, 2.8))
# left panel: LinearPulse starting at 1 ns with a 1 ns flat top
axs[0].plot(ts*1e9, [LinearPulse(t, t0=1e-9, width=1e-9) for t in ts])
axs[0].set(title="LinearPulse (x Rnorm = nTron resistance)", xlabel="time [ns]")
# right panel: OutputPulse starting at 1 ns; `op` holds the curve so we can find its peak
op = [OutputPulse(t, t0=1e-9) for t in ts]
axs[1].plot(ts*1e9, op)
# red dotted line = the peak value (~0.61)
axs[1].axhline(max(op), ls=":", color="r")
axs[1].set(title=f"OutputPulse (peak {max(op):.2f})", xlabel="time [ns]")
# lay out and show
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. The wiring: `delay` → `connectivity` → `input_connections`
#
# **Rows are senders, columns are receivers.** A nonzero entry is a connection with that delay in ns;
# 0 means no connection. Rows/columns 0–1 are the two sensors (i1, i2), 2–7 are transponders t1–t6.

# %%
# vs original: `delay` is CHANGED (see below); names, connectivity, input_delays, input_connections same
#              as the script's logic. Row = sender, column = receiver.
# [new] labels for the 8 rows/columns of the tables: 0-1 are the sensors, 2-7 are transponders t1-t6
names = ['i1', 'i2', 't1', 't2', 't3', 't4', 't5', 't6']
# STEP 1 - the whole network in one table (orig l.79-91). Row = sender, column = receiver, entry = delay in ns (0 = no link)
##                       to: i1  i2  t1  t2  t3  t4  t5  t6    from:
# vs original: the script's matrix is purely feedforward (t1->t3, t2->t3, t3->t4; t4-t6 send nothing).
# Here: t1,t2,t3 are mutually recurrent, t3->t4 is the one feedforward link, t4,t5,t6 are recurrent.
# [orig l.81-89] CHANGED wiring (see above). Values are in ns, multiplied by 1e-9 to give seconds
delay = 1.0e-9* np.array([[  0 , 0 , 5 , 0 , 0 , 0 , 9 , 0],   # i1
                          [  0 , 0 , 0 , 5 , 0 , 0 , 0 , 5],   # i2
                          [  0 , 0 , 0 , 5 , 5 , 0 , 0 , 0],   # t1  -> t2, t3
                          [  0 , 0 , 5 , 0 , 5 , 0 , 0 , 0],   # t2  -> t1, t3
                          [  0 , 0 , 5 , 5 , 0 , 5 , 0 , 0],   # t3  -> t1, t2 ; t4 (only feedforward link)
                          [  0 , 0 , 0 , 0 , 0 , 0 , 5 , 5],   # t4  -> t5, t6
                          [  0 , 0 , 0 , 0 , 0 , 5 , 0 , 5],   # t5  -> t4, t6
                          [  0 , 0 , 0 , 0 , 0 , 5 , 5 , 0],   # t6  -> t4, t5
                          ])
# [orig l.97] how many rows = how many nodes (8)
N_transponders = len(delay)
# [orig l.92] same table with 1 where a link exists, else 0 (np.ceil turns any positive delay into 1)
connectivity = np.minimum(1, np.ceil(delay))   # 1 where there is a connection, else 0
# [new] print it (the original prints the table shape at l.94-95 instead)
print("connectivity (rows = from, cols = to):\n", connectivity.astype(int))

# %% [markdown]
# The simulation needs the opposite view — for each receiver, who sends to it — so the script
# transposes. `input_delays[i][j]` is the delay of the j → i link.
#
# The **row sums** of `connectivity` are also important: a node's output is split evenly between
# everything it feeds, so a node that feeds 3 others sends each one a smaller pulse than a node
# that feeds 2.

# %%
# STEP 2 - flip the view: 'who sends to me?' (orig l.143-149)
# [orig l.145] transpose so row = receiver, column = sender
input_delays = delay.T
# [orig l.144] one empty list per receiver; will hold the node numbers that send to it
input_connections = [[] for _ in range(N_transponders)]
# [orig l.146] for each receiver i (a row of the flipped table) ...
for i, arow in enumerate(connectivity.T):
  # [orig l.147] ... look at every possible sender j
  for j, acol in enumerate(arow):
    # [orig l.148] a link exists
    if acol == 1:
      # [orig l.149] remember that j sends to i
      input_connections[i].append(j)

# [new] STEP 3 - print the wiring in words (not in the original)
for i in range(N_transponders):
  # fan-out = how many nodes this one sends to (the n that divides the pulse in RHS step 3)
  fan_out = int(np.sum(connectivity[i]))
  # print 'tX receives from [...] and sends to N node(s)'
  print(f"{names[i]}: receives from {[names[j] for j in input_connections[i]]!s:22}  "
        f"sends to {fan_out} node(s)")

# %%
# Draw the network. Grey arrows = sensor inputs, black = recurrent, red = the feedforward link.
# [new] STEP - draw the network. `pos` = x,y position of each node on the picture
pos = {0: (0, 1.5), 1: (0, -1.5), 2: (2, 1), 3: (2, -1), 4: (3.2, 0),
       5: (5, 0), 6: (6.2, 1), 7: (6.2, -1)}
# new figure
fig, ax = plt.subplots(figsize=(9, 3.6))
# for every sender j and receiver i ...
for j in range(N_transponders):
  for i in range(N_transponders):
    # ... that are connected: draw an arrow
    if connectivity[j, i]:
      # bidir = True if the link also goes the other way (then the arrow is curved so both are visible)
      bidir = connectivity[i, j] == 1
      # colour: grey = sensor input, red = the one feedforward link (t3 -> t4), black = recurrent
      col = "0.6" if j < 2 else ("r" if (j, i) == (4, 5) else "k")
      # the arrow from sender to receiver
      ax.annotate("", xy=pos[i], xytext=pos[j],
                  arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, shrinkA=14, shrinkB=14,
                                  connectionstyle=f"arc3,rad={0.18 if bidir else 0.0}"))
      # midpoint of the arrow ...
      mx, my = (pos[i][0]+pos[j][0])/2, (pos[i][1]+pos[j][1])/2
      # ... where we write the delay in ns
      ax.text(mx, my, f"{delay[j, i]*1e9:.0f}", fontsize=7, color=col, ha="center", va="center",
              bbox=dict(fc="white", ec="none", pad=0.5))
# draw each node as a circle with its name
for n, (x, y) in pos.items():
  ax.scatter(x, y, s=600, c="w", edgecolors="k", zorder=3)
  ax.text(x, y, names[n], ha="center", va="center", zorder=4)
# title, hide axes, equal aspect
ax.set_title("network (numbers = delay in ns)"); ax.axis("off"); ax.set_aspect("equal")
# show the figure
plt.show()

# %% [markdown]
# ## 4. One transponder on its own (not in the script)
#
# Each transponder has two state variables, which `solve_ivp` integrates:
#
# * **`i_nw`** — current through the *input* nTron channel. It sits at the input bias `Ib1`.
# * **`i_loop`** — current in the transponder loop, which drives the *output* nTron.
#
# How the model's names map onto the single-transponder schematic (`single_transponder_simple.pdf`):
#
# | schematic | model | notes |
# |---|---|---|
# | pulser1 (U1), pulser2 (U4) | sensors i1, i2 (`sensor_pulses`) | |
# | ① input loop: R28/L18, R29/L19 | gate current, `amp * OutputPulse` | pulse shape fixed by `OutputPulse`, not by R28/L18 |
# | bias3 in, `I_b1` | `Ib1`, `transponder_bias1` | 150 µA in LTspice, 12 µA here |
# | U3 input nTron (Q1): `L_nw1`, `R_nw1(t)` | `Lnw`, `Rnw` = `Rnorm * LinearPulse`, threshold `Inw1_sw` | current through it = `i_nw` |
# | ② shunt: L1 (5 nH), R3 (3 Ω) = `L_s1`, `R_s1` | `Ls1` = 5 nH, `Rs1` = 2 Ω | R3 differs |
# | ③ storage loop: Lb1 (~148 nH) = `L_T`, R2 (4 Ω) = `R_T` | `LT` = 150 nH, `RT0` = 3 Ω | R2 differs; current = `i_loop` (`I_loop`) |
# | bias4 in, `I_b2` | `Ib2`, `transponder_bias2` | 160 µA in LTspice, 19 µA here |
# | U2 output nTron (Q2): `L_nw2`, `R_nw2(t)` | threshold `Inw2_sw`; switching added to R2 as `RT = RT0 + Rnorm*LinearPulse` | no state variable, `L_nw2` not modelled |
# | ④ output loop: L2 (5 nH), R1 (3 Ω) = `L_out`, `R_s2` | `split` (`transponder_current_divider`) | not modelled as parts; `split` is the share that reaches the next gates |
#
# Switching is a threshold rule layered on top of the circuit equations:
#
# 1. **Input click:** when `i_nw + gate current > I1sw`, the input channel goes normal
#    (`Rnw = Rnorm * LinearPulse`). That diverts `i_nw` into the loop: `i_nw` drops to ~0 and
#    `i_loop` rises.
# 2. **Output click (the spike):** when `i_loop + Ib2 > I2sw`, the output nTron switches
#    (`RT = RT0 + Rnorm * LinearPulse`), dumping the loop current. This is the event that sends a
#    pulse to downstream nodes.
#
# `y1` and `y2` below are the circuit equations from the script's `RHS`, copied unchanged.
#
# Below, one node (input bias 12 µA) gets a single input pulse at 5 ns. A pulse peaking at
# 2.9 µA — what one spike delivers when the sender feeds 2 nodes (0.61 × 19 × 0.5 / 2) — gives
# 12 + 2.9 > 14 and the node fires; a 1.5 µA pulse doesn't.

# %%
# vs original: NEW section (not in the script). A stand-alone one-transponder copy of the rules,
#              firing each nTron at most once. y1/y2 are copied from RHS.
# [new] STEP 1 - constants for the one-transponder demo
# PEAK = highest value OutputPulse reaches (read off the plot in section 2)
PEAK = 0.6063   # max of OutputPulse (section 2)

# [new] STEP 2 - a one-transponder simulator: a copy of RHS for ONE node fed by ONE pulse, each nTron allowed to fire once
# pulse_peak = peak gate current delivered (loop 1); Ib1_n, Ib2_n = biases; t_pulse = when the pulse arrives
def single_node(pulse_peak, Ib1_n=12e-6, Ib2_n=19e-6, t_pulse=5e-9, t_end=40e-9):
  # nominal amplitude that gives the requested peak (OutputPulse only reaches 0.61 of its amplitude)
  pulse_amp = pulse_peak/PEAK   # nominal amplitude that gives this peak
  # bookkeeping: is each nTron ready (0) or clicked (1), and when it clicked
  st = dict(in_state=0, out_state=0, t_in=None, t_out=None)
  # (unused leftovers: empty lists)
  log = dict(t=[], gate=[], Rnw=[], RT=[])
  # STEP 3 - what the solver calls at every time step: given now (t) and the two currents (x), return their rates of change
  def rhs(t, x):
    # unpack the two state variables: U3 channel current i_nw and storage-loop current i_loop
    inw, iloop = x
    # loop 1: the gate current from the upstream pulse (0 until just before the pulse arrives)
    gate = pulse_amp * OutputPulse(t, t0=t_pulse) if t >= t_pulse - 2e-9 else 0.0   # loop 1 (pulser -> U3 gate)
    # ---- U2 (copy of orig l.177-197): has the transponder fired? ----
    # output nTron (U2): same rule as RHS
    # U2 is currently switched:
    if st["out_state"] == 1:
      # [orig l.185] R2 plus U2's switched resistance (1000 ohm pulse, 2 ns): this dumps the loop current
      RT = RT0 + Rnorm*LinearPulse(t, t0=st["t_out"], width=2.0e-9)
      # [orig l.187-188] after ref_period (3 ns) U2 is ready again
      if t - st["t_out"] > ref_period: st["out_state"] = 0
    # U2 not switched:
    else:
      # [orig l.191] loop 3 resistance is just R2
      RT = RT0
      # [orig l.193] firing test: storage-loop current + bias4 beats U2's threshold -> U2 fires (once only here)
      if iloop + Ib2_n > Inw2_sw and st["t_out"] is None:
        # [orig l.196-197] mark fired and record the time = the spike
        st["out_state"], st["t_out"] = 1, t
    # ---- U3 (copy of orig l.201-236): did enough gate current arrive? ----
    # input nTron (U3): same rule as RHS
    # [orig l.203] U3 starts superconducting (zero resistance)
    Rnw = 0.0
    # [orig l.220] click test: U3 channel current + gate current beats U3's threshold -> U3 clicks (once only here)
    if st["in_state"] == 0 and inw + gate > Inw1_sw and st["t_in"] is None:
      # [orig l.222-223] mark clicked and record the time
      st["in_state"], st["t_in"] = 1, t
    # U3 is in its clicked state:
    if st["in_state"] == 1:
      # [orig l.229-232] after 3 ns it recovers ...
      if t - st["t_in"] > ref_period: st["in_state"] = 0
      # [orig l.236] ... until then it has a resistance pulse (1000 ohm, ~2.5 ns): this pushes i_nw into the other branches
      else: Rnw = Rnorm*LinearPulse(t, t0=st["t_in"], width=1.0e-9)
    # [orig l.239] the incoming bias current (bias3)
    i1 = Ib1_n
    # [orig l.246] circuit equations (copied unchanged): rate of change of i_nw (U3 branch)
    y1 = (-inw*Ls1*Rnw - i1*LT*Rs1 + iloop*LT*Rs1 + inw*LT*(Rnw + Rs1) + iloop*Ls1*RT)/(Lnw*(Ls1 - LT) - Ls1*LT)
    # [orig l.247] rate of change of i_loop (storage loop 3)
    y2 = (-inw*Ls1*Rnw - i1*Lnw*Rs1 + iloop*Lnw*Rs1 + inw*Lnw*Rs1 + iloop*Lnw*RT + iloop*Ls1*RT)/(Lnw*(Ls1 - LT) - Ls1*LT)
    # [orig l.248-250] hand both rates back to the solver
    return [y1, y2]
  # STEP 4 - run it: 800 output times from 0 to t_end
  tt = np.linspace(0, t_end, 800)
  # start with U3 carrying all of Ib1 and an empty storage loop; step in <= 1 ps so no switching event is skipped (orig l.276)
  sol = solve_ivp(rhs, (0, t_end), [Ib1_n, 0.0], t_eval=tt, max_step=1e-12)
  # rebuild the gate current over time for plotting
  gate = np.array([pulse_amp*OutputPulse(t, t0=t_pulse) for t in tt])
  # give back the solution, the gate current and the click times
  return sol, gate, st

# [new] STEP 5 - run two cases and plot: a 2.9 uA pulse (fires) vs a 1.5 uA pulse (does not)
fig, axs = plt.subplots(2, 2, figsize=(10, 5), sharex=True)
# one column per pulse size
for col, amp in enumerate([2.9e-6, 1.5e-6]):
  # simulate
  sol, gate, st = single_node(amp)
  # time in ns for the x axis
  tn = sol.t*1e9
  # top panel: input side
  ax = axs[0, col]
  # grey = U3 channel current i_nw (uA)
  ax.plot(tn, sol.y[0]*1e6, color="0.5", label="i_nw")
  # black = i_nw + gate current: this is what is compared with the threshold
  ax.plot(tn, (sol.y[0]+gate)*1e6, "k", label="i_nw + gate")
  # red dotted = U3 threshold (I1sw)
  ax.axhline(Inw1_sw*1e6, color="r", ls=":", label="I1sw")
  # labels
  ax.set(title=f"input pulse peaking at {amp*1e6:.1f} uA", ylabel="input side [uA]")
  # bottom panel: loop side
  ax = axs[1, col]
  # red = storage-loop current i_loop
  ax.plot(tn, sol.y[1]*1e6, "tab:red", label="i_loop")
  # red dotted = how much i_loop is needed to fire U2 (I2sw minus Ib2)
  ax.axhline((Inw2_sw - 19e-6)*1e6, color="r", ls=":", label="I2sw - Ib2")
  # labels
  ax.set(xlabel="time [ns]", ylabel="loop [uA]")
  # mark the input click (top) and the spike (bottom) with vertical lines
  for a, tclick, lab in [(axs[0, col], st["t_in"], "input click"), (axs[1, col], st["t_out"], "output click (spike)")]:
    if tclick: a.axvline(tclick*1e9, color="tab:blue", alpha=.5, label=lab)
  # legends
  axs[0, col].legend(fontsize=7); axs[1, col].legend(fontsize=7)
  # print the click and spike times
  print(f"pulse {amp*1e6:.1f} uA: input click at {st['t_in'] and round(st['t_in']*1e9, 2)} ns, "
        f"spike at {st['t_out'] and round(st['t_out']*1e9, 2)} ns")
# show
plt.tight_layout(); plt.show()

# %% [markdown]
# What to notice: the spike comes about **7 ns after** the input click — the time the loop current
# takes to climb to `I2sw - Ib2` (1 µA here). That lag depends on the biases; in the network run
# below, t3 (input bias 11.2 µA) takes ~10 ns. And after clicking, `i_nw` takes ~20 ns to recover to
# its bias — during that time the node needs a much bigger input to click again.
#
# (This single-node version fires each nTron at most once, to keep it simple; the network `RHS`
# below lets them re-fire.)

# %% [markdown]
# ## 5. The full network
#
# ### 5a. `reset_state()` — run this before every solve
#
# `RHS` keeps its bookkeeping in global variables and **changes them while the solver runs**: it
# appends to `input_click_times` / `output_click_times`, flips `input_states` / `output_states`, and
# removes sensor clicks from `sensor_pulses` once they have fired. If you solve twice without
# resetting, the second run starts with the first run's spikes already recorded and the sensors
# already used up — no error, just wrong results. `reset_state()` rebuilds all of it from the
# parameters, which is what running the script fresh does.

# %%
# vs original: the script runs this block once at import (top level, using args.*). Here it is a
#              function so the network can be re-run. Same initial values; bias overrides come from
#              ib1_overrides / ib2_overrides dicts instead of args.ib1 / args.ib2.
# STEP 1 - function that sets up (or wipes) all the bookkeeping. Run before every solve.
# [new wrapper] in the original these are top-level lines (orig l.99-140) that run once
def reset_state():
  # these names live outside the function; `global` lets us overwrite them
  global transponder_bias1, transponder_bias2, input_states, output_states
  global input_click_times, output_click_times, sensor_pulses
  global transponder_Rnw_input, transponder_RT_loop, transponder_input_gate_current
  global transponder_source_current, transponder_initial_state

  # [orig l.114] every node gets the same input bias (bias3)
  transponder_bias1 = Ib1*np.ones(N_transponders)   # I_b1 (bias3) of every node
  # [orig l.115] every node gets the same output bias (bias4)
  transponder_bias2 = Ib2*np.ones(N_transponders)   # I_b2 (bias4) of every node
  # [orig l.117-120] then apply the per-node exceptions (orig used the --ib1 flag)
  for n, v in ib1_overrides.items(): transponder_bias1[int(n)] = v*1.0e-6
  # [orig l.121-123] same for the output bias (--ib2 flag)
  for n, v in ib2_overrides.items(): transponder_bias2[int(n)] = v*1.0e-6

  # STEP 2 - nothing has fired yet
  # state: 0 = superconducting (ready), 1 = clicked
  # [orig l.100] every U3 is ready (0 = ready, 1 = clicked)
  input_states  = np.zeros(N_transponders)
  # [orig l.101] every U2 is ready
  output_states = np.zeros(N_transponders)
  # [orig l.103] empty history of input clicks, one list per node
  input_click_times  = [[] for _ in range(N_transponders)]
  # [orig l.104] empty history of spikes, one list per node
  output_click_times = [[] for _ in range(N_transponders)]
  # [orig l.112] when each sensor will click: i1 at 1 ns, i2 at 1 ns + dt (pulsers U1 / U4)
  sensor_pulses = [[1.0e-9], [1.0e-9 + dt]]   # i1 clicks at 1 ns, i2 at 1 ns + dt
                                              #   schematic: pulser1 (U1) and pulser2 (U4)

  # STEP 3 - resistances / currents that RHS updates every step
  transponder_Rnw_input = np.zeros(N_transponders)   # U3 channel resistance "R_nw1(t)": 0 = superconducting
  # [orig l.130] loop 3 resistance starts at R2
  transponder_RT_loop   = RT0*np.ones(N_transponders)   # loop-3 resistance "R_T(t)": R2, plus U2 while it switches
  # [orig l.131] record of the gate current (for plotting only)
  transponder_input_gate_current = np.zeros(N_transponders)
  # [orig l.132] record of i_nw + gate current (for plotting only)
  transponder_source_current     = np.zeros(N_transponders)

  # STEP 4 - the starting values handed to the solver
  # solver state vector is interleaved: [i_nw(0), i_loop(0), i_nw(1), i_loop(1), ...]
  # [orig l.138] one slot per current: 2 per node = 16 slots
  transponder_initial_state = np.empty(2*N_transponders)
  # [orig l.139] even slots: every U3 starts carrying its whole bias (i_nw = Ib1)
  transponder_initial_state[0::2] = transponder_bias1   # i_nw starts at its bias
  # [orig l.140] odd slots: every storage loop starts empty (i_loop = 0)
  transponder_initial_state[1::2] = 0.0                 # no loop current yet

# STEP 5 - run it now and print what it set up
reset_state()
# bias3 of each node (uA); the overrides show up here
print("input biases  [uA]:", transponder_bias1*1e6)
print("output biases [uA]:", transponder_bias2*1e6)
print("initial state vector [uA]:", np.round(transponder_initial_state*1e6, 2))

# %% [markdown]
# ### 5b. `RHS` — copied from the script (only its progress prints were trimmed)
#
# `solve_ivp` calls this at every (sub)step with the current time `t` and state `x`. In order it:
#
# 1. fires the sensors at their scheduled times (they are just forced output clicks);
# 2. for each node's **output** nTron: sets `RT` from the last spike, or checks
#    `i_loop + Ib2 > I2sw` and records a spike;
# 3. for each node's **input** nTron: sums the pulses from every upstream spike (the *gate
#    current*), checks `i_nw + gate > I1sw` and records an input click, and sets `Rnw`;
# 4. returns `d(i_nw)/dt` and `d(i_loop)/dt` for every node from the circuit equations.
#
# The gate current from step 3 is used and then discarded — it is not part of the solution. That
# is why section 6 rebuilds it.

# %%
# vs original: identical logic. Removed: debug prints; `click_delay` temp variable (orig l.213-214; the notebook inlines it).
#              Added: comments only.
# [orig l.159] THE ENGINE. The solver calls this at every tiny time step with the time (t) and all 16 currents (x),
# and gets back how fast each current is changing. Everything below is evaluated at that instant.
def RHS(t, x):
  # [orig l.162] even entries of x: U3 channel current of each node
  nw_currents   = x[::2]    # current through U3's channel (drain -> source), loop 2
  # [orig l.163] odd entries: storage-loop current of each node
  loop_currents = x[1::2]   # "I_loop": current around the storage loop 3 (Lb1, R2) into U2's gate
  # [orig l.165] 'close enough' for matching clock times: 10 ps
  eps = 1.0e-11

  # ===== STEP 1 - the sensors fire (orig l.167-175): the outside world pokes the network =====
  # 1. sensor clicks  (schematic: pulser1 / pulser2)
  # [orig l.167] for each sensor and ...
  for n, clicks in enumerate(sensor_pulses):
    # [orig l.168] ... each click time it still has scheduled
    for i_click, sensor_click_t in enumerate(clicks):
      # [orig l.169] is it that time now (within 10 ps)?
      if abs(t - sensor_click_t) <= eps:
        # (orig l.170 prints different debug text)
        print(f"sensor {names[n]} clicks at {t*1e9:.2f} ns")
        # [orig l.171] mark the sensor 'clicked'
        output_states[n] = 1
        # [orig l.174] RECORD THE CLICK TIME: every node listening to this sensor will find it in STEP 3
        output_click_times[n].append(t)
        # [orig l.175] remove it from the schedule so it fires once
        sensor_pulses[n].pop(i_click)

  # ===== STEP 2 - output nTron U2 (orig l.177-197): did each transponder fire? =====
  # 2. output nTrons  (schematic: U2). U2 itself has no state variable here: when it switches, its
  #    resistance is added onto R2 as RT, which dumps I_loop. L_nw2, L2 and R1 are not modelled.
  # [orig l.178] for each node
  for i_out, output_state in enumerate(output_states):
    # [orig l.179] U2 is currently switched (fired recently):
    if output_states[i_out] == 1:
      # [orig l.183] when did it last fire?
      tlast = output_click_times[i_out][len(output_click_times[i_out])-1]
      # [orig l.185] while switched: loop resistance = R2 + a 1000 ohm pulse (2 ns). This dumps the stored loop current
      transponder_RT_loop[i_out] = RT0 + Rnorm*LinearPulse(t, t0=tlast, width=2.0e-9)   # R2 + U2 switched
      # [orig l.187] after ref_period (3 ns) ...
      if t - tlast > ref_period:
        # [orig l.188] ... U2 may fire again
        output_states[i_out] = 0
    # [orig l.190] U2 is ready:
    if output_states[i_out] == 0:
      # [orig l.191] loop resistance is just R2
      transponder_RT_loop[i_out] = RT0
      # [orig l.193] FIRING TEST: storage-loop current (U2's gate) + bias4 > U2's threshold?
      if loop_currents[i_out] + transponder_bias2[i_out] > Inw2_sw:   # I_loop (into U2 gate) + I_b2 > U2 threshold
        # [orig l.196] yes: U2 switches
        output_states[i_out] = 1
        # [orig l.197] RECORD THE SPIKE TIME. This is the moment current leaves the transponder
        output_click_times[i_out].append(t)

  # ===== STEP 3 - input nTron U3 (orig l.199-236): did enough arrive from upstream? =====
  # 3. input nTrons  (schematic: U3). The gate current is loop 1: in the schematic it comes from the
  #    pulsers through R28/L18 and R29/L19; here its shape is fixed by OutputPulse instead.
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
        # [orig l.208] how many nodes that sender feeds (fan-out)
        n_split = np.sum(connectivity[j_in])
        # [orig l.209] the sender's output bias (bias4)
        b2  = transponder_bias2[j_in]
        # [orig l.210] PULSE SIZE from this sender = Ib2 x split / fan-out (the only use of `split`)
        amp = b2*transponder_current_divider/n_split   # sender's I_b2 x split, shared between its targets
        # [orig l.212] for every spike that sender has ever fired ...
        for click_t in output_click_times[j_in]:
          # [orig l.213-214] ... when does it arrive here? spike time + link delay (the delay box)
          click_arrival = click_t + input_delays[i_in][j_in]
          # [orig l.215] add that pulse's current right now to the gate current (loop 1). Pulses from several senders add up
          ntron_gate_current += amp*OutputPulse(t, t0=click_arrival)   # loop 1: current into U3's gate
      # [orig l.218] (record i_nw + gate, for plotting)
      transponder_source_current[i_in]     = nw_currents[i_in] + ntron_gate_current
      # [orig l.219] (record gate current, for plotting)
      transponder_input_gate_current[i_in] = ntron_gate_current
      # [orig l.220] CLICK TEST: U3 channel current + gate current > U3's threshold?
      if nw_currents[i_in] + ntron_gate_current > Inw1_sw:   # U3 channel + gate current > U3 threshold
        # [orig l.222] yes: U3 clicks
        input_states[i_in] = 1
        # [orig l.223] record the click time
        input_click_times[i_in].append(t)
    # [orig l.226] U3 is clicked (resistive):
    if input_states[i_in] == 1:
      # [orig l.228] when did it click?
      latest_input_click = input_click_times[i_in][len(input_click_times[i_in])-1]
      # [orig l.229] after 3 ns ...
      if t - latest_input_click > ref_period:
        # [orig l.232] ... U3 is ready again
        input_states[i_in] = 0
      # [orig l.234] otherwise still resistive:
      else:
        # [orig l.236] U3's resistance: 1000 ohm trapezoid (about 2.5 ns). This pushes current out of U3 into the other branches
        transponder_Rnw_input[i_in] = Rnorm*LinearPulse(t, t0=latest_input_click, width=1.0e-9)   # U3 "R_nw1(t)"

  # ===== STEP 4 - circuit physics (orig l.237-250): where does the current go? =====
  # 4. circuit equations: I_b1 splits between three parallel branches to ground --
  #    U3 (Lnw, Rnw), shunt L1/R3 (Ls1, Rs1) and storage loop Lb1/R2 (LT, RT).
  #    y1 = d(i_nw)/dt, y2 = d(I_loop)/dt. NOTE: these give the right steady-state split, but the
  #    Lnw*Ls1 term in the denominator has the opposite sign from the circuit as drawn
  #    (drawn: Lnw*Ls1 + Lnw*LT + Ls1*LT); time constants differ by ~5%. Worth checking with Whit.
  # [orig l.237] list of rates of change we will return
  y = []
  # [orig l.238] for each node
  for i_node, loop_cur in enumerate(loop_currents):
    # [orig l.239] its input bias (bias3) flowing in
    i1  = transponder_bias1[i_node]   # I_b1 (bias3)
    # [orig l.240] current in U3 right now
    inw = nw_currents[i_node]
    # [orig l.241] current in the storage loop right now
    iloop = loop_cur
    # [orig l.242] U3's resistance right now (from STEP 3)
    Rnw = transponder_Rnw_input[i_node]   # U3 "R_nw1(t)"
    # [orig l.243] loop 3's resistance right now (from STEP 2)
    RT  = transponder_RT_loop[i_node]     # "R_T(t)" = R2 (+ U2 while switching)
    # [orig l.246] d(i_nw)/dt: how fast the U3 current changes (Kirchhoff + inductors)
    y1 = (-inw*Ls1*Rnw - i1*LT*Rs1 + iloop*LT*Rs1 + inw*LT*(Rnw + Rs1) + iloop*Ls1*RT)/(Lnw*(Ls1 - LT) - Ls1*LT)
    # [orig l.247] d(i_loop)/dt: how fast the storage-loop current changes
    y2 = (-inw*Ls1*Rnw - i1*Lnw*Rs1 + iloop*Lnw*Rs1 + inw*Lnw*Rs1 + iloop*Lnw*RT + iloop*Ls1*RT)/(Lnw*(Ls1 - LT) - Ls1*LT)
    # [orig l.248-249] store both
    y.append(y1)
    y.append(y2)
  # [orig l.250] hand all 16 rates back to the solver, which takes the next step and calls RHS again
  return y

# %% [markdown]
# ### 5c. Solve (about a minute)
#
# `max_step = 1 ps` is what makes this slow; it is needed so the solver doesn't step over the
# sub-nanosecond switching events.

# %%
# vs original: the script calls solve_ivp inside main() with 100e-9 written in; here it is a function
#              using T_END, and calls reset_state() first.
# STEP 1 - run the whole simulation (wrapper is new; the original does this inside main(), orig l.254-276)
def solve_network():
  # wipe all bookkeeping so a second run starts clean
  reset_state()
  # [orig l.271] the 1000 times at which results are saved
  t_pts = np.linspace(0, T_END, num=1000)
  # [orig l.276] start at 0, run to T_END, step <= 1 ps so no sub-ns switching event is skipped (this is why it takes a minute)
  return solve_ivp(RHS, (0, T_END), transponder_initial_state, t_eval=t_pts, max_step=1.0e-12)

# STEP 2 - run it
solution = solve_network()
# time axis in seconds and in ns
t, tns = solution.t, solution.t/1.0e-9
# status 0 = finished OK; shape (16, 1000) = 16 currents at 1000 times
print("solver status:", solution.status, "| state array shape:", solution.y.shape)
# STEP 3 - print every node's spike times (orig only prints debug lines)
print("\noutput (spike) click times [ns]:")
# for every node (sensors and transponders) ...
for n in range(N_transponders):
  # ... print its spike times in ns (rounded to 0.01)
  print(f"  {names[n]}: {np.round(np.array(output_click_times[n])/1.0e-9, 2)}")
# ... and every node's input-click times
print("\ninput (gate) click times [ns]:")
# for the transponders only (sensors have no input nTron) ...
for n in range(2, N_transponders):
  # ... print its input-click times in ns
  print(f"  {names[n]}: {np.round(np.array(input_click_times[n])/1.0e-9, 2)}")

# %% [markdown]
# ### 5d. The script's original trace plot
#
# `solution.y` is 16 rows: row `2n` is `i_nw` of node n, row `2n+1` is `i_loop`. The script's
# 6-panel figure plots rows 4–9, i.e. **t1, t2, t3** — its labels "gate 2/3/4" are row indices,
# and "gate" there is the channel current `i_nw`, not the gate current.

# %%
# [orig l.278] STEP - the original 6-panel plot: i_nw (top of each pair) and i_loop (red) for t1, t2, t3
fig, axs = plt.subplots(6, sharex=True, figsize=(8, 7))
# [orig l.280-285] for t1, t2, t3 (node numbers 2, 3, 4) ...
for k, n in enumerate([2, 3, 4]):
  # ... plot i_nw (row 2n of the solution) in uA
  axs[2*k].plot(tns, solution.y[2*n]/1e-6, label=f"i_nw {names[n]}")
  # ... and i_loop (row 2n+1) in red
  axs[2*k+1].plot(tns, solution.y[2*n+1]/1e-6, "r", label=f"i_loop {names[n]}")
# [orig l.293-298, 300-301] legend and axis labels
for ax in axs:
  ax.legend(loc="right", fontsize=7); ax.set_ylabel("uA")
# time label on the bottom panel
axs[-1].set_xlabel("time [ns]")
# show (original saves to ../results instead, l.307-308)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. Rebuilding the gate current
#
# Same formula as step 3 of `RHS`, applied after the run to the recorded spike times: every spike
# from j reaches i after the j → i delay and adds `amp * OutputPulse`, where
# `amp = Ib2(j) * split / (number of nodes j feeds)`. `gate_parts[i][j]` keeps each upstream node's
# contribution separately; `gate_current[i]` is their sum.

# %%
# vs original: NEW (post-processing added to the script after the run; not in main). Same recipe
#              as RHS step 3 (orig l.207-215), applied to the recorded click times.
# [new] STEP 1 - rebuild the gate current after the run (RHS threw it away). Same recipe as RHS step 3 (orig l.207-215)
# gate_parts[i][j] = the pulse node i received from node j; gate_current[i] = their sum
gate_parts = [dict() for _ in range(N_transponders)]
# one row per node, one column per saved time
gate_current = np.zeros((N_transponders, len(t)))
# for each receiver ...
for i_in in range(N_transponders):
  # ... and each sender feeding it
  for j_in in input_connections[i_in]:
    # pulse size = sender's Ib2 x split / sender's fan-out (orig l.208-210)
    amp  = transponder_bias2[j_in]*transponder_current_divider/np.sum(connectivity[j_in])
    # this sender's contribution over time, starts at 0
    part = np.zeros(len(t))
    # for each spike the sender fired ...
    for click_t in output_click_times[j_in]:
      # ... it arrives after the link delay (orig l.213-214)
      arrival = click_t + input_delays[i_in][j_in]
      # add the pulse at every saved time, but only after the spike happened (RHS cannot see the future)
      part += np.array([amp*OutputPulse(tk, t0=arrival) if tk >= click_t else 0.0 for tk in t])
    # keep this sender's piece
    gate_parts[i_in][j_in] = part
    # and add it to the total
    gate_current[i_in] += part

# %% [markdown]
# Pick a node and see exactly which spikes land on it, when, and how big. Change `node` to inspect
# another one (0 = i1 … 7 = t6).

# %%
# vs original: NEW (notebook only). `node` picks which transponder to inspect.
# [new] STEP 2 - inspect one node. Change this number (0 = i1 ... 7 = t6)
node = 4   # t3
# gate current this node needs: threshold minus its input bias (uA)
need = (Inw1_sw - transponder_bias1[node])*1e6
# print node name, bias, and the gate current it needs
print(f"{names[node]}: input bias {transponder_bias1[node]*1e6:.1f} uA -> needs {need:.2f} uA of gate current to click\n")
# table header
print(f"{'from':>5} {'spike [ns]':>11} {'arrives [ns]':>13} {'peak [uA]':>10}")
# each sender ...
for j_in in input_connections[node]:
  # ... its pulse size
  amp = transponder_bias2[j_in]*transponder_current_divider/np.sum(connectivity[j_in])
  # ... each of its spikes
  for click_t in output_click_times[j_in]:
    # ... arrival time here
    arr = click_t + input_delays[node][j_in]
    # one row: sender, spike time, arrival time, peak size (0.606 x nominal amplitude)
    print(f"{names[j_in]:>5} {click_t*1e9:11.2f} {arr*1e9:13.2f} {0.606*amp*1e6:10.2f}")
# the biggest total gate current this node ever saw
print(f"\nlargest summed gate current on {names[node]}: {gate_current[node].max()*1e6:.2f} uA")

# %% [markdown]
# ## 7. The three plots
#
# ### 7a. Spike raster
# Black ticks = output clicks (spikes); red triangles = input (gate) clicks. Dashed lines separate
# sensors | t1–t3 | t4–t6.

# %%
# vs original: NEW (the script has an equivalent raster added after main; not in main).
# [new] STEP 1 - spike raster: one row per node, one tick per spike
fig, ax = plt.subplots(figsize=(8, 3.5))
# for each node:
for n in range(N_transponders):
  # black tick at each spike time
  ax.vlines(np.array(output_click_times[n])/1.0e-9, n-0.4, n+0.4, color='k', lw=2,
            label='output click (spike)' if n == 0 else None)
  # red triangle at each input (gate) click
  if input_click_times[n]:
    ax.plot(np.array(input_click_times[n])/1.0e-9, [n-0.45]*len(input_click_times[n]), 'v',
            color='tab:red', ms=4, label='input (gate) click' if not any(input_click_times[:n]) else None)
# dashed separators: sensors | t1-t3 | t4-t6
for y in (1.5, 4.5):
  ax.axhline(y, color='0.8', ls='--', lw=0.8)
# label rows with node names
ax.set_yticks(range(N_transponders), names)
# i1 at the top
ax.set_ylim(N_transponders-0.5, -0.5)
# axes and title
ax.set(xlim=(0, tns[-1]), xlabel='time [ns]', title='spike raster')
# legend
ax.legend(loc='upper right', fontsize=8, frameon=False)
# show
plt.show()

# %% [markdown]
# ### 7b. Gate-current trace for one node
# Top: each upstream contribution (dashed) and their sum (black) against the gate current needed to
# click. Bottom: what the threshold rule actually compares — `i_nw + gate` against `I1sw`.
# Uses `node` from section 6.

# %%
# vs original: NEW (the script has an equivalent gate-trace plot added after main; not in main).
# [new] STEP 1 - gate-current trace for the node chosen in section 6
n = node
# this node's U3 channel current over time (row 2n of the solution)
inw = solution.y[2*n]
# two stacked panels
fig, axs2 = plt.subplots(2, sharex=True, figsize=(8, 5))
# TOP: each sender's pulse (dashed) ...
for j_in, part in gate_parts[n].items():
  axs2[0].plot(tns, part/1.0e-6, '--', label=f'from {names[j_in]}')
# ... and their sum (black)
axs2[0].plot(tns, gate_current[n]/1.0e-6, 'k', lw=1.5, label='summed gate current')
# red dotted: gate current needed to click (threshold minus bias)
axs2[0].axhline((Inw1_sw - transponder_bias1[n])/1.0e-6, color='r', ls=':', label='needed to switch (I1sw - Ib1)')
# labels
axs2[0].set(ylabel='gate current [uA]', title=f'{names[n]}: gate current')
# legend
axs2[0].legend(fontsize=8, frameon=False)
# BOTTOM: grey = U3 channel current ...
axs2[1].plot(tns, inw/1.0e-6, color='0.5', label='channel current i_nw')
# ... black = channel current + gate current: what the click test compares (orig l.220)
axs2[1].plot(tns, (inw + gate_current[n])/1.0e-6, 'k', label='i_nw + gate')
# red dotted = U3's threshold
axs2[1].axhline(Inw1_sw/1.0e-6, color='r', ls=':', label='I1sw (threshold)')
# faint red vertical lines at this node's input clicks
for k, tc in enumerate(input_click_times[n]):
  axs2[1].axvline(tc/1.0e-9, color='r', alpha=0.4, label='input click' if k == 0 else None)
# labels
axs2[1].set(xlabel='time [ns]', ylabel='current [uA]')
# legend
axs2[1].legend(fontsize=8, frameon=False)
# show
plt.show()

# %% [markdown]
# ### 7c. Gradient raster
# Grey = gate current (white = none, black = the largest gate current in the run). Each row goes
# white for `ref_period` after an input click. Solid red = spike, dotted red = input click.

# %%
# vs original: NEW (added to the script after main; the original has only the 6-panel trace plot).
# [new] STEP 1 - gradient raster: shade = gate current (white = none, black = the biggest in the run)
def gradient_raster(ax, gate_current, input_click_times, output_click_times, title=None):
  # the largest gate current anywhere (avoid dividing by 0)
  gmax  = gate_current.max() if gate_current.max() > 0 else 1.0
  # scale every value to 0..1
  shade = gate_current/gmax
  # for each node ...
  for n in range(N_transponders):
    # ... and each input click ...
    for tc in input_click_times[n]:
      # ... paint white for ref_period afterwards (the node is recovering, so it does not 'charge up')
      shade[n][(t > tc) & (t <= tc + ref_period)] = 0
  # draw the shading as an image (rows = nodes, columns = time)
  im = ax.imshow(shade, aspect='auto', cmap='Greys', vmin=0, vmax=1, interpolation='nearest',
                 extent=(tns[0], tns[-1], N_transponders-0.5, -0.5))
  # STEP 2 - overlay the clicks
  for n in range(N_transponders):
    # solid red = spike
    ax.vlines(np.array(output_click_times[n])/1.0e-9, n-0.5, n+0.5, color='red', lw=1.8,
              label='output click (spike)' if n == 0 else None)
    # dotted red = input click
    ax.vlines(np.array(input_click_times[n])/1.0e-9, n-0.5, n+0.5, color='red', lw=0.8, ls=':',
              label='input (gate) click' if n == 2 else None)
  # separators
  for y in (1.5, 4.5):
    ax.axhline(y, color='0.8', ls='--', lw=0.8)
  # row labels
  ax.set_yticks(range(N_transponders), names)
  # axis label and title
  ax.set(xlabel='time [ns]', title=title or 'gradient raster: gate current (white = 0, black = max)')
  # give back the image and the scale
  return im, gmax

# STEP 3 - make the figure with this run's data
fig, ax = plt.subplots(figsize=(8, 3.5))
# draw it
im, gmax = gradient_raster(ax, gate_current.copy(), input_click_times, output_click_times)
# colour bar: shade -> fraction of the largest gate current
fig.colorbar(im, ax=ax, label=f'gate current / {gmax/1.0e-6:.2f} uA', pad=0.01)
# legend below the plot
ax.legend(loc='upper center', fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.15))
# show
plt.show()

# %% [markdown]
# ## 8. Bias sweep
#
# Vary one node's input bias and keep everything else fixed. With the defaults: t3 (node 4) needs
# ~3.0 µA of gate current at 11 µA and a single pulse brings ~2.9 µA, so the interesting range is
# around 11 µA. Each value is a full solve (~1 min). Edit `sweep_node` / `sweep_values` to taste.

# %%
# vs original: NEW (notebook only). Re-runs the whole network once per input bias value.
# [new] STEP 1 - choose what to sweep: node 4 (= t3) and the input biases to try (uA)
sweep_node   = 4
sweep_values = [10.5, 11.0, 11.2, 12.0]   # uA

# STEP 2 - remember the current overrides so we can restore them
saved_overrides = dict(ib1_overrides)
# results: bias value -> that run's click times
sweep = {}
# for each bias value:
for v in sweep_values:
  # set that node's input bias (reset_state reads this dict, orig l.117-120 equivalent)
  ib1_overrides[sweep_node] = v
  # run the whole network (about a minute)
  sol = solve_network()
  # copy this run's spike and click times (the lists are reused by the next run)
  sweep[v] = dict(out=[list(c) for c in output_click_times], inp=[list(c) for c in input_click_times])
  # print the number of spikes of each node
  print(f"{names[sweep_node]} Ib1 = {v:5.1f} uA -> spikes per node:",
        {names[k]: len(c) for k, c in enumerate(output_click_times) if k >= 2})
# put the original overrides back
ib1_overrides.clear(); ib1_overrides.update(saved_overrides)   # put the parameter back

# %%
# STEP 3 - one raster per bias value
fig, axs = plt.subplots(len(sweep_values), sharex=True, figsize=(8, 1.9*len(sweep_values)))
# for each value:
for ax, v in zip(axs, sweep_values):
  # each node: ticks at its spikes (swept node in red)
  for n in range(N_transponders):
    ax.vlines(np.array(sweep[v]["out"][n])/1e-9, n-0.4, n+0.4, color='r' if n == sweep_node else 'k', lw=1.6)
  # separators
  for y in (1.5, 4.5):
    ax.axhline(y, color='0.8', ls='--', lw=0.8)
  # labels
  ax.set_yticks(range(N_transponders), names, fontsize=7)
  # i1 at top
  ax.set_ylim(N_transponders-0.5, -0.5)
  # title = bias value
  ax.set_title(f"{names[sweep_node]} input bias = {v} uA", fontsize=9)
# x axis
axs[-1].set(xlim=(0, T_END*1e9), xlabel="time [ns]")
# show
plt.tight_layout(); plt.show()

# %% [markdown]
# With the default settings, t3 (red) goes from silent → one spike → two spikes as its bias rises,
# but **t4–t6 are identical in every run**: t3's spike never makes t4 fire, because t4 already
# clicked at ~22 ns — driven by t5 and t6, which get the sensor signals directly (i1 → t5,
# i2 → t6) — and is still recovering when t3's pulse arrives. To
# make the t3 → t4 link matter, set `delay[0, 6]` and `delay[1, 7]` to 0 in section 3.
#
# **Before re-running anything above:** sections 5–7 use the global `solution` and click lists from
# the last `solve_network()` call. The sweep overwrote the click lists, so re-run section 5c before
# going back to sections 6–7.
