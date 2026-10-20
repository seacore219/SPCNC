# Notebook N06, section by section and line by line — for a non-coding audience

Line numbers refer to `notebooks/S06_toy_example_walkthrough.py` (the paired `.py` of `N06_toy_example_walkthrough.ipynb`).
Part names refer to the single-transponder schematic (`single_transponder_simple.pdf`) and the six-copy
figure `docs/figures/N06_network_schematic.png`.

## Vocabulary you need (3 ideas)

* **nTron** = a superconducting wire with a "gate". Superconducting = zero resistance. If the current
  through the wire exceeds a **switching current** (the threshold), it flips to a resistive state for a
  few nanoseconds, then recovers. A little extra current injected at the gate lowers how much current the
  wire itself needs. Each transponder has two: **U3** (input) and **U2** (output).
* **A "click"** = the moment an nTron flips resistive. U3 clicking = "the neuron received enough input".
  U2 clicking = "the neuron fired" (a spike).
* **The program is a stopwatch.** The solver walks through time in 1-picosecond steps. At every step it
  calls `RHS(t, x)` once: "it is now time `t`, the circuit currents are `x`; what is happening, and how
  fast is each current changing?" Everything below runs again at every step.

The two numbers tracked per transponder (the vector `x`):

| name in code | meaning | where in the schematic |
|---|---|---|
| `i_nw` (`nw_currents`) | current flowing through U3's channel | the vertical U3 wire, bias3 → ground |
| `i_loop` (`loop_currents`) | current circulating in the storage loop | loop ③ (through Lb1 and R2) |

## How close is the notebook to the real script?

Checked in this session.

* **Same results.** I ran the original `toy_example.py` with `--dt 35 --Ib1init 12 --ib1 2 13 --ib1 4 11.2 --Ib2init 19 --I1sw 14 --split 0.5`. Every spike time matches the notebook's to the printed 0.01 ns (t1: 10.75, 29.69, 50.77; t3: 26.25; t6: 28.77, 45.62, 69.00; and so on).
* **Same physics, line for line.** I diffed `RHS`, `LinearPulse`, `OutputPulse` and the constants (`Ls1`, `Rs1`, `LT`, `RT0`, `Lnw`, `Rnorm`, `ref_period`) against the script. The only differences are removed debug `print`s, spacing, and two cosmetic temporary variables (`click_delay`, `delta_t`).
* **What differs on purpose:**

| | script | notebook |
|---|---|---|
| settings | command-line flags, defaults 16 / 16 / 20 / 20 / split 0.1 / dt 5 ns | editable variables, set to 12 / 19 / 14 / 20 / 0.5 / 35 ns |
| run length | 100 ns written into the code | `T_END` variable |
| setup | runs once when the file starts | wrapped in `reset_state()` so you can re-run |
| time-of-flight generator (`--generator`) | supported | not included |
| sections 4 and 8 | not in the script | added (single-node demo, bias sweep) |
| section 4 `single_node` | — | a simplified stand-alone copy of the rules in which each nTron can fire only once |
| plots (sections 5d, 7) | built inside `main()` and saved to files | the same plots, shown inline |

Note: a comment in the script says `ref_period` is "also not used??". It is used, in `RHS` at l.402 and l.431 of the notebook.

---

## Section 1 — Parameters (l.38–112): "the knobs"

Plain-language rule: these are the numbers you can change. Everything else follows.

| line(s) | variable | what it is | schematic |
|---|---|---|---|
| `dt_ns` | gap between the two sensor clicks | when pulser2 fires after pulser1 | pulsers U1 / U4 |
| `Ib1init` | input bias, all nodes (µA) | current fed into U3's drain; the closer to U3's threshold, the less extra push it needs | **bias3 / I_b1** |
| `Ib2init` | output bias, all nodes (µA) | current fed into U2's drain; also sets the size of the pulse a spike sends on | **bias4 / I_b2** |
| `I1sw_uA` | U3 switching current | threshold of the input nTron | **U3** |
| `I2sw_uA` | U2 switching current | threshold of the output nTron | **U2** |
| `split` | fraction of a spike sent to the next gates | only used at l.420 | **output junction L2 / R1** |
| `ib1_overrides`, `ib2_overrides` | per-node exceptions | node numbers: 0 = i1, 1 = i2, 2 = t1 … 7 = t6 | bias3 / bias4 of one box |
| `T_END` | how long to simulate | — | — |

The next block converts µA to A (`Inw1_sw`, `Inw2_sw`, `Ib1`, `Ib2`, `dt`), then sets the fixed circuit values:

| variable | value | meaning | schematic |
|---|---|---|---|
| `Ls1`, `Rs1` | 5 nH, 2 Ω | shunt branch around U3 | **L1, R3** (loop ②) |
| `LT`, `RT0` | 150 nH, 3 Ω | storage loop | **Lb1, R2** (loop ③) |
| `Lnw` | 10 nH | inductance of the nTron channel | inside U3 / U2 |
| `Rnorm` | 1000 Ω | resistance of an nTron while it is switched | U3 / U2 when clicked |
| `ref_period` | 3 ns | rest time after a click before the same nTron may click again | — |

## Section 2 — The two pulse shapes (l.114–153)

Think of these as two stencils the program reuses.

| function | shape | used for | schematic |
|---|---|---|---|
| `LinearPulse(t, t0, width, rise, fall)` | a trapezoid between 0 and 1: ramps up for `rise`, stays at 1 for `width`, ramps down for `fall` | multiplied by `Rnorm` to give an nTron's resistance while it is switched | U3 or U2 turning resistive |
| `OutputPulse(t, t0, tau, mu0)` | a smooth step that switches on around `t0` (width `mu0` = 0.2 ns), then decays with time constant `tau` = 1 ns; peaks at **0.606** | the current an upstream spike pushes into this node's gate | loop ① (R_in, L_in) |

The `LinearPulse` lines: `if (t-t0 < 0) or (...)` → before the pulse or after it, return 0; the next
`if` → ramp up; the next → ramp down; the final `else` → flat top, return 1.
The `OutputPulse` line is one formula: (smooth step) × (exponential decay).
Because the step is smooth rather than sudden, the pulse already delivers about 20 % of the nominal amplitude 0.35 ns
*before* `t0`. That is why t1's first input click (5.64 ns) comes slightly before the nominal arrival
(0.99 + 5 = 5.99 ns): t1 only needs 1 µA of gate current, which the early part of the pulse supplies.

The two small plots at the end of the cell just draw the shapes.

## Section 3 — The wiring (l.155–218)

| line(s) | what it does |
|---|---|
| `names = [...]` | labels for the 8 rows: sensors i1, i2, then t1–t6 |
| `delay = 1e-9 * np.array(...)` | **the whole network in one table.** Row = sender, column = receiver, entry = delay in ns (0 = no link). Row t3 reads "sends to t1, t2 (5 ns) and t4 (5 ns, the one feedforward link)" |
| `connectivity = np.minimum(1, np.ceil(delay))` | same table with 1 where a link exists, else 0 |
| `input_delays = delay.T` | flip the table so we can ask "who sends to me?" |
| `for i, arow in enumerate(connectivity.T): ... input_connections[i].append(j)` | for each receiver, build the list of senders (e.g. t1 listens to i1, t2, t3) |
| `fan_out = np.sum(connectivity[i])` | how many nodes each sender feeds; this is the `n` that divides the pulse in l.420 |
| the plotting cell | draws the network (circles = nodes, arrows = links, numbers = delay in ns, red = the feedforward link) |

Schematic: each arrow is one wire with a delay box (the 5 ns boxes in the six-copy figure).

## Section 4 — One transponder on its own (l.220–318): "the smallest working example"

This section is not in the script. It simulates a single transponder fed by one pulse, so you can see the cause-and-effect chain without the network.

| lines | what it does | schematic |
|---|---|---|
| `PEAK = 0.6063` | the peak of `OutputPulse`, used to convert "a pulse peaking at X µA" into the code's nominal amplitude | — |
| `pulse_amp = pulse_peak/PEAK` | scale so the pulse peaks at the requested µA | loop ① |
| `gate = pulse_amp * OutputPulse(...)` | the gate current at this instant | loop ① → U3's gate |
| the "output nTron" block | same firing rule as l.406, except it can only fire once | U2 |
| the "input nTron" block | same click rule as l.426, once only | U3 |
| `y1`, `y2` | the circuit equations, identical to `RHS` | loops ②, ③ |
| `solve_ivp(rhs, ..., [Ib1_n, 0.0], ...)` | start with U3 carrying all of its bias and the storage loop empty, then let time run | — |
| the 2×2 plot | left column: a 2.9 µA pulse → click then spike. Right column: a 1.5 µA pulse → nothing | — |

The take-home: a pulse of 2.9 µA pushes `i_nw + gate` above the 14 µA threshold (12 + 2.9). U3 clicks
at 4.9 ns. About 7 ns later `i_loop` has climbed to 1 µA (`Inw2_sw − Ib2` = 20 − 19) and U2 fires
(spike at 11.9 ns). A 1.5 µA pulse never gets over the threshold.

## Section 5a — `reset_state()` (l.322–365)

The simulation keeps a notebook of what has happened (click times, who is "ready"). Before every run it must be wiped clean, or a second run starts with the first run's spikes already in it.

| lines | what it does |
|---|---|
| `transponder_bias1 = Ib1 * np.ones(N)` and the `for n, v in ib1_overrides.items()` loop | give every node the same input bias, then apply the per-node exceptions |
| same two lines for `bias2` | same, for the output bias |
| `input_states`, `output_states` = zeros | every nTron starts ready (0 = ready, 1 = clicked) |
| `input_click_times`, `output_click_times` = empty lists | no clicks recorded yet |
| `sensor_pulses = [[1e-9], [1e-9 + dt]]` | schedule i1 at 1 ns and i2 at 1 ns + `dt` |
| `transponder_Rnw_input = 0` | U3 starts superconducting (no resistance) |
| `transponder_RT_loop = RT0` | loop ③ starts with just R2 |
| `transponder_initial_state[0::2] = bias1`, `[1::2] = 0` | starting currents: U3 carries its whole bias, the storage loop is empty |

---

## Section 5b — `RHS` (l.367–453; the function starts at l.382): the engine

### Setup lines (l.382–385)

| line | what it does | schematic |
|---|---|---|
| 382 `def RHS(t, x)` | "Given the time and the currents, work out what happens next." | — |
| 383 `nw_currents = x[::2]` | pull out every transponder's U3 channel current (every other number in `x`) | U3 wire |
| 384 `loop_currents = x[1::2]` | pull out every transponder's storage-loop current | loop ③ |
| 385 `eps = 1e-11` | "close enough" window for matching clock times (10 ps) | — |

### Step 1 — the two sensors (l.387–394): the outside world pokes the network

Sensors i1 and i2 stand in for the pulsers (U1, U4) on the left of the schematic.

| line | what it does |
|---|---|
| 388–389 | go through each sensor and each click time it still has scheduled (i1 at 1 ns, i2 at 1 ns + `dt`) |
| 390 | is the clock now (almost exactly) at that click time? |
| 392 | mark the sensor as "clicked" |
| 393 | **write the click time into the record** — every transponder that listens to this sensor will read it in Step 3 |
| 394 | cross it off the to-do list so it fires once |

### Step 2 — the output nTron U2 (l.396–408): "did this transponder fire?"

U2 has no current of its own in the model. Its effect is that when it flips resistive, it adds resistance to loop ③ and drains the stored current.

| line | what it does | schematic |
|---|---|---|
| 398–399 | for each transponder: has it fired recently? | U2 |
| 400 | look up when it last fired | — |
| 401 | while firing, loop ③'s resistance is R2 plus a pulse of 1000 Ω for ~2 ns. That resistance dumps the stored current | R2 + U2 switching |
| 402–403 | after `ref_period` (3 ns) it is allowed to fire again | refractory period |
| 404–405 | if not firing, loop ③'s resistance is just R2 (`RT0`) | R2 |
| **406** | **the firing test: `i_loop + Ib2 > Inw2_sw`.** Storage-loop current feeds U2's gate; bias4 (`Ib2`) runs through U2's drain; if the sum beats U2's threshold, U2 flips | gate of U2 + bias4 |
| 407–408 | mark it fired and record the time. **This is the spike**: the only moment current "leaves" the transponder | output loop ④ |

### Step 3 — the input nTron U3 (l.410–434): "did enough arrive from upstream?"

| line | what it does | schematic |
|---|---|---|
| 412–413 | for each transponder whose U3 is ready (not already clicked) | U3 |
| 414–416 | start with no resistance and no gate current | — |
| 417 | go through every transponder that sends to this one | links / delay boxes |
| 418 | count how many places that sender feeds (its fan-out) | output junction |
| 419 | the sender's output bias (`Ib2`) | bias4 of the sender |
| **420** | **`amp = Ib2 × split / n`**: the size of the pulse this sender gives each receiver. This is the only place `split` is used | output junction L2/R1 |
| 421–422 | for every spike the sender has ever fired, work out when it arrives: spike time + link delay | delay box |
| 423 | add that pulse's current (shape from `OutputPulse`) to this node's gate current. Several pulses add up (coincidence detection) | loop ① (R_in, L_in) |
| 424–425 | store the totals so we can plot them later | — |
| **426** | **the input test: `i_nw + gate current > Inw1_sw`.** If U3's own current plus the gate boost beats its threshold, U3 flips | U3 |
| 427–428 | mark it clicked and record the time | — |
| 429–432 | after 3 ns it may click again | refractory period |
| 433–434 | while clicked, U3 has a resistance pulse of up to 1000 Ω (about 2.5 ns). This forces current out of U3 and into the other branches | U3 resistive |

### Step 4 — the circuit physics (l.436–451): "where does the current go?"

This is ordinary circuit law (Kirchhoff plus inductors): the bias current `Ib1` coming in at bias3 splits
between three parallel paths to ground, and inductors resist sudden change, so currents change gradually.

| line | what it does | schematic |
|---|---|---|
| 442 | for each transponder... | — |
| 443 | `i1` = the incoming bias current | bias3 / `Ib1` |
| 444–445 | read this transponder's two current numbers | U3 wire, loop ③ |
| 446 | `Rnw` = U3's resistance right now (0 unless it just clicked, from l.434) | U3 |
| 447 | `RT` = loop ③'s resistance right now (R2, plus U2 when it fires, from l.401) | R2 + U2 |
| 448 | `y1` = how fast U3's channel current is changing | U3 branch |
| 449 | `y2` = how fast the storage-loop current is changing | loop ③ |
| 450–452 | return those rates to the solver, which takes the next 1 ps step and calls `RHS` again | — |

Why a click leads to a spike a few ns later: when U3 turns resistive (l.434), current is pushed out of
U3 into the shunt (L1/R3) and the storage loop (Lb1/R2). The inductor makes `i_loop` rise slowly. When
`i_loop + Ib2` passes U2's threshold (l.406), U2 fires.


---

## Section 5c — Solve (l.455–475)

| line | what it does |
|---|---|
| `reset_state()` | wipe the notebook clean |
| `t_pts = np.linspace(0, T_END, num=1000)` | the 1,000 moments at which results are saved |
| `solve_ivp(RHS, (0, T_END), initial_state, t_eval=t_pts, max_step=1e-12)` | the solver: start at 0, run to `T_END`, ask `RHS` for rates, never step more than 1 ps. The 1 ps cap is why it takes about a minute; without it the solver could step right over a sub-nanosecond switching event |
| the print loops | list each node's spike times and input-click times |

## Section 5d — The script's original plot (l.477–492)

`solution.y` holds 16 rows: row 2n is `i_nw` of node n, row 2n+1 is `i_loop`. The plot shows these for t1, t2, t3. The sawtooth shape is: current sits at the bias, drops when U3 clicks, recovers over ~20 ns.

## Section 6 — Rebuilding the gate current (l.494–529)

During the run, the gate current was computed and thrown away (it is not part of the solution). Here we recompute it from the recorded spike times, using the same recipe as l.417–423.

| line | what it does |
|---|---|
| `for i_in ... for j_in in input_connections[i_in]` | for every receiver and each of its senders |
| `amp = bias2[j] * split / np.sum(connectivity[j])` | the pulse size from this sender (the `split` line) |
| `arrival = click_t + input_delays[i][j]` | when the spike gets there |
| `part += amp*OutputPulse(tk, t0=arrival) if tk >= click_t` | add the pulse, but only after the spike happened |
| `gate_parts[i][j] = part`, `gate_current[i] += part` | keep each sender's piece, and the total |

The next cell (`node = 4`) prints, for one node, every spike that reaches it: when it was sent, when it
arrives, and its peak size. For t3 each pulse is 2.88 µA and t3 needs 2.80 µA, so one pulse is just
enough.

## Section 7 — The three plots (l.531–608)

| plot | what the code draws | how to read it |
|---|---|---|
| 7a spike raster | `ax.vlines` at each output click time; red triangles at input clicks | one row per node, a tick = a spike |
| 7b gate-current trace | top: each sender's pulse (dashed) and the sum (black) against the "needed to switch" line, `Inw1_sw − Ib1`; bottom: `i_nw + gate` against `Inw1_sw` | the click happens where the black curve in the bottom panel crosses the red dotted line |
| 7c gradient raster | `gate_current / gmax` as a grey image (white = 0, black = the biggest gate current in the run); set to white for `ref_period` after each input click; solid red lines = spikes, dotted red = input clicks | dark stretch = a neuron being driven; white = just clicked or nothing arriving |

## Section 8 — Bias sweep (l.610–652)

| line | what it does |
|---|---|
| `sweep_node = 4`, `sweep_values = [...]` | pick a node (t3) and the input biases to try |
| `ib1_overrides[sweep_node] = v` then `solve_network()` | set that bias, run the whole network again |
| `sweep[v] = dict(out=..., inp=...)` | save the spike and click times of this run |
| `ib1_overrides.update(saved_overrides)` | put the original setting back |
| the figure | one raster per bias value, t3 in red |

Result: below ~11.2 µA t3 stays silent, at 11.2 it fires once, at 12 twice. t4–t6 are the same in every
run because t4 is already driven by t5 and t6 before t3's pulse arrives.

## Where the four loops fit in the code

| loop | code | role |
|---|---|---|
| ① input | l.417–425 | gate current from upstream spikes |
| ② shunt | `Ls1`, `Rs1` in l.448–449 | catches current while U3 is resistive |
| ③ storage | `LT`, `RT` in l.448–449, `i_loop` | accumulates, drains via R2, drives U2's gate |
| ④ output | l.420 (`split`) | what each spike delivers to the next transponders |

## What the model leaves out

L2, R1, and L_nw2 are not simulated. `split` is a single number standing in for the L2/R1 current
division. The gate pulse shape is fixed (`OutputPulse`), not computed from the R_in/L_in branch.
The `y1`/`y2` formula has a sign difference in the `Lnw·Ls1` denominator term versus the drawn circuit
(about 5% on time constants); worth checking with Whit.
