# The simulation core (`RHS`), line by line — for a non-coding audience

Line numbers refer to `notebooks/S06_toy_example_walkthrough.py` (same code as `toy_example.py`).
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

## Setup lines (l.382–385)

| line | what it does | schematic |
|---|---|---|
| 382 `def RHS(t, x)` | "Given the time and the currents, work out what happens next." | — |
| 383 `nw_currents = x[::2]` | pull out every transponder's U3 channel current (every other number in `x`) | U3 wire |
| 384 `loop_currents = x[1::2]` | pull out every transponder's storage-loop current | loop ③ |
| 385 `eps = 1e-11` | "close enough" window for matching clock times (10 ps) | — |

## Step 1 — the two sensors (l.387–394): the outside world pokes the network

Sensors i1 and i2 stand in for the pulsers (U1, U4) on the left of the schematic.

| line | what it does |
|---|---|
| 388–389 | go through each sensor and each click time it still has scheduled (i1 at 1 ns, i2 at 1 ns + `dt`) |
| 390 | is the clock now (almost exactly) at that click time? |
| 392 | mark the sensor as "clicked" |
| 393 | **write the click time into the record** — every transponder that listens to this sensor will read it in Step 3 |
| 394 | cross it off the to-do list so it fires once |

## Step 2 — the output nTron U2 (l.396–408): "did this transponder fire?"

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

## Step 3 — the input nTron U3 (l.410–434): "did enough arrive from upstream?"

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

## Step 4 — the circuit physics (l.436–451): "where does the current go?"

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

## Where the loops fit

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
