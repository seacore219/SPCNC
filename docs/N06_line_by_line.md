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

Checked in this session. "Original" below means `toy_example.py` on `main` (Whit's version). The
`toy_example.py` on branch `toy-example-recurrent-rasters` is the original plus the changes you asked
for (recurrent wiring, raster, gate trace, gradient raster).

* **Same results as the branch script.** I ran the branch `toy_example.py` with `--dt 35 --Ib1init 12 --ib1 2 13 --ib1 4 11.2 --Ib2init 19 --I1sw 14 --split 0.5`. Every spike time matches the notebook's to the printed 0.01 ns (t1: 10.75, 29.69, 50.77; t3: 26.25; t6: 28.77, 45.62, 69.00; and so on).
* **Same physics as the original.** I diffed `RHS`, `LinearPulse`, `OutputPulse` and the constants against the original. The only differences are removed debug `print`s, spacing and two cosmetic temporary variables (`click_delay`, `delta_t`). I did not run the original (feedforward-only) wiring.

### Everything that differs from the original

| item | original (`main`) | notebook | why |
|---|---|---|---|
| **`delay` matrix** | feedforward only: t1→t3, t2→t3, t3→t4; t4–t6 send nothing | t1-t2-t3 recurrent, t3→t4 feedforward, t4-t5-t6 recurrent | your first request |
| `dt` | flag `--dt`, default 5 ns | variable `dt_ns` = 35 | no command line in a notebook |
| `Ib1`, `Ib2` | flags `--Ib1init`, `--Ib2init`, default 16 / 16 | variables `Ib1init` = 12, `Ib2init` = 19 | same |
| `Inw1_sw`, `Inw2_sw` | flags `--I1sw`, `--I2sw`, default 20 / 20 | variables `I1sw_uA` = 14, `I2sw_uA` = 20 | same |
| `split` | flag `--split`, default 0.1 | variable `split` = 0.5 | same |
| per-node biases | flags `--ib1 NODE VALUE`, `--ib2 NODE VALUE` (repeatable); none by default | dicts `ib1_overrides = {2: 13.0, 4: 11.2}`, `ib2_overrides = {}` | same; the two values are mine |
| run length | `100*1e-9` written twice inside `main()` | variable `T_END` | editable |
| start-up state | top-level code, runs once | `reset_state()` function | so the network can be re-run |
| time-of-flight generator | `--generator` flag, `tof_gen` import | not included | not needed here |
| `recover_time1` | defined, unused | not defined | dead variable |
| `RHS` | has debug prints, `click_delay` temp | prints removed, inlined | readability; same logic |
| `solve_ivp` call | inside `main()` | `solve_network()` function | so it can be re-run |
| 6-panel trace plot | in `main()` | section 5d, same plot | — |
| rasters, gate trace, gradient raster | not in `main` | script (branch) and sections 6–7 | your requests |
| section 4 (single transponder), section 8 (bias sweep) | not in the script | notebook only | teaching aids |

Note: a comment in the original says `ref_period` is "also not used??". It is used, in `RHS` at RHS step 2 and RHS step 3 of the notebook.

---

## Section 1 — Parameters: "the knobs"

**vs original:** the flags become variables (see the "vs original" columns below); the conversions and fixed circuit values are unchanged.

Plain-language rule: these are the numbers you can change. Everything else follows.

| variable | meaning | schematic | vs original |
|---|---|---|---|
| `dt_ns` | gap between the two sensor clicks | pulsers U1 / U4 | was flag `--dt` (default 5); now a variable, 35 |
| `Ib1init` | input bias, all nodes (µA); the closer to U3's threshold, the less extra push it needs | **bias3 / I_b1** | was `--Ib1init` (default 16); now 12 |
| `Ib2init` | output bias, all nodes (µA); also sets the size of the pulse a spike sends on | **bias4 / I_b2** | was `--Ib2init` (default 16); now 19 |
| `I1sw_uA` | U3 switching current (threshold of the input nTron) | **U3** | was `--I1sw` (default 20); now 14, renamed |
| `I2sw_uA` | U2 switching current (threshold of the output nTron) | **U2** | was `--I2sw` (default 20); same value, renamed |
| `split` | fraction of a spike sent to the next gates; only used at RHS step 3 `amp =` line | **output junction L2 / R1** | was `--split` (default 0.1); now 0.5 |
| `ib1_overrides` | per-node exceptions to the input bias: `{node number: µA}`, 2 = t1, 4 = t3 | bias3 of one box | was repeatable flag `--ib1 NODE VALUE`; now a dict; the values 13 and 11.2 are mine |
| `ib2_overrides` | per-node exceptions to the output bias | bias4 of one box | was `--ib2 NODE VALUE`; now an empty dict |
| `T_END` | how long to simulate | none | **new** variable (script hard-codes 100 ns) |

The next block converts µA to A (`Inw1_sw`, `Inw2_sw`, `Ib1`, `Ib2`, `dt`), then sets the fixed circuit values:

| variable | value | meaning | schematic | vs original |
|---|---|---|---|---|
| `Ls1`, `Rs1` | 5 nH, 2 Ω | shunt branch around U3 | **L1, R3** (loop ②) | unchanged |
| `LT`, `RT0` | 150 nH, 3 Ω | storage loop | **Lb1, R2** (loop ③) | unchanged |
| `Lnw` | 10 nH | inductance of the nTron channel | inside U3 / U2 | unchanged |
| `Rnorm` | 1000 Ω | resistance of an nTron while it is switched | U3 / U2 when clicked | unchanged |
| `ref_period` | 3 ns | rest time after a click before the same nTron may click again | — | unchanged (the script's comment "not used??" is wrong) |

## Section 2 — The two pulse shapes

**vs original:** same functions. `LinearPulse` keeps `t-t0` in a temporary variable `delta_t` in the script; the maths is the same.

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

## Section 3 — The wiring

**vs original:** the `delay` table is **changed** (recurrent wiring instead of feedforward only); `connectivity`, `input_delays` and `input_connections` are the same logic. The script also prints the matrix shape; the notebook prints the connectivity and fan-out instead.

| line(s) | what it does |
|---|---|
| `names = [...]` | labels for the 8 rows: sensors i1, i2, then t1–t6 |
| `delay = 1e-9 * np.array(...)` | **the whole network in one table.** Row = sender, column = receiver, entry = delay in ns (0 = no link). Row t3 reads "sends to t1, t2 (5 ns) and t4 (5 ns, the one feedforward link)" |
| `connectivity = np.minimum(1, np.ceil(delay))` | same table with 1 where a link exists, else 0 |
| `input_delays = delay.T` | flip the table so we can ask "who sends to me?" |
| `for i, arow in enumerate(connectivity.T): ... input_connections[i].append(j)` | for each receiver, build the list of senders (e.g. t1 listens to i1, t2, t3) |
| `fan_out = np.sum(connectivity[i])` | how many nodes each sender feeds; this is the `n` that divides the pulse in RHS step 3 `amp =` line |
| the plotting cell | draws the network (circles = nodes, arrows = links, numbers = delay in ns, red = the feedforward link) |

Schematic: each arrow is one wire with a delay box (the 5 ns boxes in the six-copy figure).

## Section 4 — One transponder on its own: "the smallest working example"

**vs original:** **new**, not in the script. A stand-alone single transponder that copies the rules (and `y1`/`y2`) from `RHS`, firing each nTron at most once.

This section is not in the script. It simulates a single transponder fed by one pulse, so you can see the cause-and-effect chain without the network.

| lines | what it does | schematic |
|---|---|---|
| `PEAK = 0.6063` | the peak of `OutputPulse`, used to convert "a pulse peaking at X µA" into the code's nominal amplitude | — |
| `pulse_amp = pulse_peak/PEAK` | scale so the pulse peaks at the requested µA | loop ① |
| `gate = pulse_amp * OutputPulse(...)` | the gate current at this instant | loop ① → U3's gate |
| the "output nTron" block | same firing rule as RHS step 2 firing test, except it can only fire once | U2 |
| the "input nTron" block | same click rule as RHS step 3 click test, once only | U3 |
| `y1`, `y2` | the circuit equations, identical to `RHS` | loops ②, ③ |
| `solve_ivp(rhs, ..., [Ib1_n, 0.0], ...)` | start with U3 carrying all of its bias and the storage loop empty, then let time run | — |
| the 2×2 plot | left column: a 2.9 µA pulse → click then spike. Right column: a 1.5 µA pulse → nothing | — |

The take-home: a pulse of 2.9 µA pushes `i_nw + gate` above the 14 µA threshold (12 + 2.9). U3 clicks
at 4.9 ns. About 7 ns later `i_loop` has climbed to 1 µA (`Inw2_sw − Ib2` = 20 − 19) and U2 fires
(spike at 11.9 ns). A 1.5 µA pulse never gets over the threshold.

## Section 5a — `reset_state()`

**vs original:** same initial values, but wrapped in a function, and per-node biases come from the `ib1_overrides` / `ib2_overrides` dicts instead of the `--ib1` / `--ib2` flags.

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

## Section 5b — `RHS`: the engine

**vs original:** identical logic. Removed: debug prints and the `click_delay` temporary (now inlined at RHS step 3). Added: comments.

**Line numbers:** a notebook has none, so elsewhere in this document and in the notebook comments, places in `RHS` are named ("RHS step 2 firing test", etc.). The "script line" columns below are line numbers in `transponder_model-whit_0918/src/tests/toy_example.py` on branch `toy-example-recurrent-rasters`. Your own copy (e.g. `toy_example_raster.py`) may differ by a few lines.

### Setup lines

| script line | what it does | schematic |
|---|---|---|
| 161 `def RHS(t, x)` | "Given the time and the currents, work out what happens next." | — |
| 164 `nw_currents = x[::2]` | pull out every transponder's U3 channel current (every other number in `x`) | U3 wire |
| 165 `loop_currents = x[1::2]` | pull out every transponder's storage-loop current | loop ③ |
| 167 `eps = 1e-11` | "close enough" window for matching clock times (10 ps) | — |

### Step 1 — the two sensors: the outside world pokes the network

Sensors i1 and i2 stand in for the pulsers (U1, U4) on the left of the schematic.

| script line | what it does |
|---|---|
| 169–170 | go through each sensor and each click time it still has scheduled (i1 at 1 ns, i2 at 1 ns + `dt`) |
| 171 | is the clock now (almost exactly) at that click time? |
| 173 | mark the sensor as "clicked" |
| 176 | **write the click time into the record** — every transponder that listens to this sensor will read it in Step 3 |
| 177 | cross it off the to-do list so it fires once |

### Step 2 — the output nTron U2: "did this transponder fire?"

U2 has no current of its own in the model. Its effect is that when it flips resistive, it adds resistance to loop ③ and drains the stored current.

| script line | what it does | schematic |
|---|---|---|
| 180–181 | for each transponder: has it fired recently? | U2 |
| 185 | look up when it last fired | — |
| 187 | while firing, loop ③'s resistance is R2 plus a pulse of 1000 Ω for ~2 ns. That resistance dumps the stored current | R2 + U2 switching |
| 189–190 | after `ref_period` (3 ns) it is allowed to fire again | refractory period |
| 192–193 | if not firing, loop ③'s resistance is just R2 (`RT0`) | R2 |
| **406** | **the firing test: `i_loop + Ib2 > Inw2_sw`.** Storage-loop current feeds U2's gate; bias4 (`Ib2`) runs through U2's drain; if the sum beats U2's threshold, U2 flips | gate of U2 + bias4 |
| 198–199 | mark it fired and record the time. **This is the spike**: the only moment current "leaves" the transponder | output loop ④ |

### Step 3 — the input nTron U3: "did enough arrive from upstream?"

| script line | what it does | schematic |
|---|---|---|
| 203–204 | for each transponder whose U3 is ready (not already clicked) | U3 |
| 205–208 | start with no resistance and no gate current | — |
| 209 | go through every transponder that sends to this one | links / delay boxes |
| 210 | count how many places that sender feeds (its fan-out) | output junction |
| 211 | the sender's output bias (`Ib2`) | bias4 of the sender |
| **420** | **`amp = Ib2 × split / n`**: the size of the pulse this sender gives each receiver. This is the only place `split` is used | output junction L2/R1 |
| 214–216 | for every spike the sender has ever fired, work out when it arrives: spike time + link delay | delay box |
| 217 | add that pulse's current (shape from `OutputPulse`) to this node's gate current. Several pulses add up (coincidence detection) | loop ① (R_in, L_in) |
| 220–221 | store the totals so we can plot them later | — |
| **426** | **the input test: `i_nw + gate current > Inw1_sw`.** If U3's own current plus the gate boost beats its threshold, U3 flips | U3 |
| 224–225 | mark it clicked and record the time | — |
| 228–234 | after 3 ns it may click again | refractory period |
| 236–238 | while clicked, U3 has a resistance pulse of up to 1000 Ω (about 2.5 ns). This forces current out of U3 and into the other branches | U3 resistive |

### Step 4 — the circuit physics: "where does the current go?"

This is ordinary circuit law (Kirchhoff plus inductors): the bias current `Ib1` coming in at bias3 splits
between three parallel paths to ground, and inductors resist sudden change, so currents change gradually.

| script line | what it does | schematic |
|---|---|---|
| 240 | for each transponder... | — |
| 241 | `i1` = the incoming bias current | bias3 / `Ib1` |
| 242–243 | read this transponder's two current numbers | U3 wire, loop ③ |
| 244 | `Rnw` = U3's resistance right now (0 unless it just clicked, from RHS step 3) | U3 |
| 245 | `RT` = loop ③'s resistance right now (R2, plus U2 when it fires, from RHS step 2) | R2 + U2 |
| 248 | `y1` = how fast U3's channel current is changing | U3 branch |
| 249 | `y2` = how fast the storage-loop current is changing | loop ③ |
| 250–252 | return those rates to the solver, which takes the next 1 ps step and calls `RHS` again | — |

Why a click leads to a spike a few ns later: when U3 turns resistive (RHS step 3), current is pushed out of
U3 into the shunt (L1/R3) and the storage loop (Lb1/R2). The inductor makes `i_loop` rise slowly. When
`i_loop + Ib2` passes U2's threshold (RHS step 2 firing test), U2 fires.


---

## Section 5c — Solve

**vs original:** same solver call; wrapped in `solve_network()`, run length from `T_END`.

| line | what it does |
|---|---|
| `reset_state()` | wipe the notebook clean |
| `t_pts = np.linspace(0, T_END, num=1000)` | the 1,000 moments at which results are saved |
| `solve_ivp(RHS, (0, T_END), initial_state, t_eval=t_pts, max_step=1e-12)` | the solver: start at 0, run to `T_END`, ask `RHS` for rates, never step more than 1 ps. The 1 ps cap is why it takes about a minute; without it the solver could step right over a sub-nanosecond switching event |
| the print loops | list each node's spike times and input-click times |

## Section 5d — The script's original plot

**vs original:** same plot as the script's `main()`.

`solution.y` holds 16 rows: row 2n is `i_nw` of node n, row 2n+1 is `i_loop`. The plot shows these for t1, t2, t3. The sawtooth shape is: current sits at the bias, drops when U3 clicks, recovers over ~20 ns.

## Section 6 — Rebuilding the gate current

**vs original:** **new** (post-processing added to the script for your gate-trace request).

During the run, the gate current was computed and thrown away (it is not part of the solution). Here we recompute it from the recorded spike times, using the same recipe as RHS step 3.

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

## Section 7 — The three plots

**vs original:** **new** (raster, gate trace, gradient raster were added on your request; the original only has the 6-panel trace plot).

| plot | what the code draws | how to read it |
|---|---|---|
| 7a spike raster | `ax.vlines` at each output click time; red triangles at input clicks | one row per node, a tick = a spike |
| 7b gate-current trace | top: each sender's pulse (dashed) and the sum (black) against the "needed to switch" line, `Inw1_sw − Ib1`; bottom: `i_nw + gate` against `Inw1_sw` | the click happens where the black curve in the bottom panel crosses the red dotted line |
| 7c gradient raster | `gate_current / gmax` as a grey image (white = 0, black = the biggest gate current in the run); set to white for `ref_period` after each input click; solid red lines = spikes, dotted red = input clicks | dark stretch = a neuron being driven; white = just clicked or nothing arriving |

## Section 8 — Bias sweep

**vs original:** **new**, notebook only.

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
| ① input | RHS step 3 | gate current from upstream spikes |
| ② shunt | `Ls1`, `Rs1` in RHS step 4 | catches current while U3 is resistive |
| ③ storage | `LT`, `RT` in RHS step 4, `i_loop` | accumulates, drains via R2, drives U2's gate |
| ④ output | RHS step 3 `amp =` line (`split`) | what each spike delivers to the next transponders |

## What the model leaves out

L2, R1, and L_nw2 are not simulated. `split` is a single number standing in for the L2/R1 current
division. The gate pulse shape is fixed (`OutputPulse`), not computed from the R_in/L_in branch.
The `y1`/`y2` formula has a sign difference in the `Lnw·Ls1` denominator term versus the drawn circuit
(about 5% on time constants); worth checking with Whit.
