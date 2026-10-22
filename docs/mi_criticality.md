# Mutual information as a proximity-to-criticality measure for the transponder network

Script: `transponder_model-whit_0918/src/tests/mi_criticality_sweep.py` (uses IDTxl network inference).
Simulator: `transponder_model-whit_0918/src/tests/toy_example.py` with new opt-in flags (defaults unchanged).

## 1. Why mutual information tracks criticality

**What the grant asks for.** Thrust 2's question is "how does proximity to dynamical criticality govern the
information processing capacity of a PWC network performing TOF measurement?" The proposal states that criticality
"maximizes dynamic range, information transmission, and sensitivity to weak inputs", and that transponder networks
can reach it because "their operating point can be adjusted via bias currents" (full_grant, sections 1 and 3.2).

**The argument, in three steps.**

1. *MI is capped by the response entropy.* Shew et al. 2011 measure information transmission as
   "MI(S;R) = H(R) - H(R|S)" and note that "H(R|S) is always positive, MI(S;R) is bounded by H(R)."
2. *Response entropy collapses on both sides of the critical point.* In the subcritical regime "activity rates drop
   sufficiently low to cause low entropy even though interactions are also low". In the supercritical regime
   "both high activity rates and strong interactions cause low entropy". Their result: "entropy and information
   transmission are maximized for the particular E/I ratio specified by κ=1", where κ = 1 is the critical
   branching point.
3. *The same peak is seen for MI between units, not just stimulus-response MI.* In the kinetic Ising model,
   "average mutual information ... peaks ... precisely at the phase transition", whereas "global transfer entropy
   peaked distinctly on the disordered side of the phase transition" (Barnett et al. 2013, PRL 111:177203, as
   summarised by the first author). This is why we use MI, not TE, as the marker of the critical point.

**The closest precedent is in the project folder.** Cramer et al. 2020 (Nat. Commun. 11:2853,
`s41467020165483 1.pdf`) tune a 32-neuron neuromorphic chip towards criticality and report: "the AIS of a neuron,
as well as the I and the transfer entropy (TE) between pairs of neurons increase with lower Kext" (that is, closer
to critical). They also report "the lagged mutual information between pairs of neurons in the reservoir also
increases", and that "A similar increase in I, AIS, and TE has been observed for the Ising model and reservoirs at
criticality". Their estimation recipe is the one used here: "We calculated H, AIS, I, and TE with the toolbox JIDT"
(JIDT is the estimator engine inside IDTxl), and "The activity is obtained by binning the spike data with δt = τref".
Two caveats from the same paper: "this maximal capacity is typically not necessary; ... it can even be
unfavorable when solving simple tasks", and their small network was only ever "as close to a critical state as
possible". A 6-transponder network cannot have a true phase transition either; what we can find is the crossover
where MI peaks.

**Earlier evidence.** Beggs & Plenz 2003 (J. Neurosci. 23:11167) found a branching ratio "close to unity" in cortical
slices, and "showed that this branching ratio optimizes information transmission in feedforward networks models".
That quote is secondhand, from Chialvo's review arXiv:cond-mat/0402538.

## 2. What this means for this network

- **Control parameter: `Ib1`, the input bias.** It is the knob the grant names and the one that can be turned on a
  fabricated chip. (`split` is fixed by the inductor ratios at fabrication, but the sweep can vary it with
  `--param split`.)
- **Why `Ib1` works as a gain knob.** A transponder fires on one upstream spike when the pulse peak,
  0.61 x Ib2 x split / n_out, exceeds the gap I1sw - Ib1. Raising `Ib1` shrinks that gap. Each spike then triggers
  more downstream spikes, so the branching ratio rises.
- **Subcritical (low `Ib1` or low `split`).** Sensor clicks die out after zero or one hops. Transponders are
  nearly silent, so H ≈ 0 and MI ≈ 0.
- **Supercritical (high `split`).** Each spike triggers more than one downstream spike, so the recurrent loops keep
  reverberating on their own. Firing stops tracking the sensors, and I_in collapses.
- **Near critical (m ≈ 1).** Sensor events propagate through t1-t3 → t4 → t4-t6 without dying out or being
  swamped. Stimulus information should be largest here.
- **`Ib1` alone cannot reach the supercritical side.** In the pilot it does not get there (section 5). After a spike,
  the storage-loop current takes L/R ≈ 50 ns to recover, and that blocks re-excitation in the loops, so m saturates
  at about 0.8. Coupling strength (`split`) is what pushes m past 1. On hardware, `Ib2` scales the same pulse
  amplitude and can be tuned after fabrication.

**Two measures, both from IDTxl `BivariateMI.analyse_network` with the discrete JIDT estimator.** The targets are
t1 to t6, and the source lags run from 1 to 5 bins (3 to 15 ns), which covers one or two hops. Only MI that
passes IDTxl's surrogate tests counts; anything non-significant is set to 0.

| measure | sources for each target | literature analogue |
|---|---|---|
| `I_in` | the two sensors i1, i2 | stimulus-response MI (Shew 2011); lagged input-neuron I (Cramer 2020, Fig. 7a) |
| `I_int` | the other five transponders | pairwise / lagged MI between units (Barnett 2013; Cramer 2020, Figs. 7c, 8b) |

**Use `I_in` as the criticality proxy, not `I_int`.** The pilot shows why. In the Ising model the ordered phase is
frozen, so pairwise MI falls past the transition. Here the supercritical phase is self-sustained reverberation,
which is itself highly predictable from node to node, so `I_int` keeps rising past m = 1. `I_int / H` peaks only
weakly.

Both are also reported divided by H, as Cramer 2020 do "to compensate for potential changes in the firing rate".

**Independent check: the branching ratio m.** m is the mean number of downstream transponders that spike 1–8 ns
after a transponder's pulse arrives. The wiring and delays are read from `toy_example.py` itself. m = 1 is the
classical critical point, and m is also the Thrust 2 milestone metric (M2.2). The usual multistep-regression
estimator (Wilting & Priesemann 2018) was tried and rejected. Population autocorrelation in this network swings
negative one generation after a spike (the storage loop's L/R ≈ 50 ns recovery suppresses the node), so the
exponential fit r_k = b·m^k that the estimator relies on does not hold.

## 3. Changes to `toy_example.py` (all opt-in; the defaults reproduce the original)

| flag | default | why |
|---|---|---|
| `--rate` | 0 (off) | Poisson sensor clicks instead of the 2 fixed clicks. MI needs many independent input events, and this is the "driven" setting of Shew 2011 / Cramer 2020. |
| `--sw_noise` | 0 (off) | Gaussian noise (µA) on U3's switching current, redrawn each time U3 resets. Without it every link either always or never transmits, so m jumps from 0 to n_out and there is no graded branching to tune. Nanowire switching currents are physically stochastic. |
| `--tmax`, `--max_step` | 100 ns, 1 ps | Run length and solver step (see the speed note below). |
| `--seed`, `--save_spikes`, `--no_plots` | | Reproducibility and output for the sweep. |

Two more changes:

- **`OutputPulse` overflow fix.** It is rewritten as `1/(1+e^-a)`, which is algebraically identical to the original
  `1 - 1/(e^a+1)`. The original crashed with `OverflowError` once a spike was more than ~140 ns old, so runs
  longer than ~150 ns were impossible.
- **Wiring unchanged.** The recurrent t1-t2-t3 → t4 → t4-t5-t6 configuration works for MI. Per-node values in
  `summary_*.json` let you compare the groups: t1–t3 are entries 0–2 and t4–t6 are entries 3–5.

**Speed vs. accuracy.** 100 ns takes about 50 s at `--max_step 1e-3` (1 ps, the original) and about 12 s at `5e-3`.
At 5 ps most spike times move by less than 0.6 ns, but a marginal coincidence can flip (in the Whit-parameter run,
t3's spike moved from 26.25 to 15.85 ns). That happens because RHS records a click at the solver's trial stages.
The pilot below uses 5 ps. For production on a cluster, use `--fixed "... --max_step 1e-3"`.

## 4. How to run

```bash
pip install Cython jpype1 statsmodels h5py networkx
pip install --no-build-isolation git+https://github.com/pwollstadt/IDTxl.git   # needs Java (JDK >= 11)
cd transponder_model-whit_0918/src/tests
export PYTHONPATH=..:../..
# 1) Ib1: silent -> active (subcritical side only)
python mi_criticality_sweep.py --values 11.5 12 12.5 13 13.3 13.6 13.9 14.2 --trials 12
# 2) split at Ib1 = 13: crosses m = 1 (this is the sweep that shows the MI peak)
python mi_criticality_sweep.py --param split --values 0.3 0.45 0.6 0.7 0.8 0.9 1.0 --trials 10 \
    --fixed "--Ib1init 13 --Ib2init 19 --I1sw 14 --split 0.5 --rate 0.033 --sw_noise 0.3 --max_step 5e-3"
```

- **IDTxl and numpy 2.** IDTxl still calls `np.math`, which numpy 2 removed. The script restores it with
  `np.math = math`.
- **Outputs.** `../results/mi_sweep/summary_<param>.json` and `mi_sweep_<param>.png`.
- **Resuming.** Each simulation is cached as an `.npz`, so an interrupted sweep resumes where it stopped.
- **Data size.** Each value uses `--trials` × `--tmax`/3 ns samples per node, as IDTxl replications. The
  surrogate tests permute replications, so use at least 8 trials.

## 5. Pilot results (5 ps step, 300 ns runs, Poisson rate 0.033 /ns per sensor, sw_noise 0.3 µA)

**Sweep 2: `split` at Ib1 = 13 (10 runs per value)** (`figures/mi_sweep_split.png`, `figures/summary_split.json`)

| split | 0.3 | 0.45 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|
| m (descendants / spike) | 0.62 | 0.77 | 0.93 | 1.09 | 1.29 | 1.34 | 1.29 |
| rate (spikes/ns/node) | 0.019 | 0.029 | 0.039 | 0.049 | 0.064 | 0.071 | 0.074 |
| **I_in** (bits) | 0.054 | **0.071** | 0.065 | 0.034 | 0.007 | 0.000 | 0.001 |
| I_in / H | 0.17 | 0.17 | 0.13 | 0.06 | 0.01 | 0 | 0.00 |
| I_int (bits) | 0.072 | 0.126 | 0.114 | 0.116 | 0.167 | 0.172 | 0.120 |
| I_int / H | 0.23 | **0.30** | 0.22 | 0.19 | 0.24 | 0.23 | 0.16 |

- **I_in is an inverted U.** It peaks at split 0.45–0.6 (m ≈ 0.8–0.9). As m crosses 1 it falls by ~10x, and at
  m ≈ 1.3 the network carries no significant information about its input. This is the Shew 2011 prediction.
- **The peak sits slightly below m = 1.** That is expected for a small, externally driven network (compare
  Priesemann et al. 2014, "Spike avalanches in vivo suggest a driven, slightly subcritical brain state", ref. 67 of
  Cramer 2020). With 10 runs the grid cannot separate 0.45 from 0.6, so refine 0.4–0.75 in production.

**Sweep 1: `Ib1` at split 0.5 (12 runs per value)** (`figures/mi_sweep_Ib1init.png`, `figures/summary_Ib1init.json`)

| Ib1 (µA) | 11 | 11.5 | 12 | 12.5 | 13 | 13.3 | 13.6 | 13.9 | 14.2 |
|---|---|---|---|---|---|---|---|---|---|
| m | 0 | 0.55 | 0.75 | 0.77 | 0.79 | 0.70 | 0.62 | 0.59 | 0.70 |
| I_in (bits) | 0 | 0.022 | 0.045 | 0.069 | 0.077 | 0.092 | 0.086 | 0.111 | 0.084 |
| I_int (bits) | 0 | 0.039 | 0.063 | 0.095 | 0.111 | 0.138 | 0.146 | 0.136 | 0.142 |

(Ib1 = 11 comes from a 6-run pilot.)

- **The network goes from silent to active.** MI climbs from 0 and then plateaus, but m never reaches 1, so this
  sweep covers only the subcritical side.
- **m falls again above 13.** Noise-driven spontaneous clicks, which are not caused by a parent spike, dilute the
  descendants count.

**Reading the numbers.** The MI values are small in absolute terms (≤ 0.17 bits/bin), because spikes are sparse
(H ≈ 0.3–0.7 bits). Only MI that passes IDTxl's surrogate tests is counted. Reruns of the IDTxl step move the
values by about ±0.002 bits, because the surrogates are random.

## 6. Caveats

- **No true phase transition.** Six transponders give a crossover, not a phase transition. Finite-size scaling
  (Cramer 2020, Fig. 3) needs larger networks; the grant asks for ≥ 10 transponders.
- **The pilot used a 5 ps step.** Rerun the final sweep at 1 ps (`--max_step 1e-3`) on a cluster and check that the
  peak stays put.
- **m is approximate.** Descendants are counted with a 1–8 ns window after pulse arrival. A spike driven by two
  coincident parents is credited to both, so m is slightly overestimated near and above 1.
- **`--sw_noise` sets how sharp the transition is.** At 0.3 µA it is a modelling choice. Thrust 3 switching-current
  histograms (M3.2) should replace it with the measured width.
