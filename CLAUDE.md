# transponder-reduction

## What this project is
Part of the DOE Genesis Mission project "Superconducting Polychronous Computation Near
Criticality" (Argonne / MIT / NIST / WashU). This repo is the WashU (Hengen lab, Thrust 2)
side-project that reduces the ANL/MIT SPICE-level nTron transponder circuit model down to a
fast dynamical-systems neuron model, so it can be dropped into the network-scale PWC
simulator (which cannot afford to run full SPICE per-node for a >=10-transponder network).

Goal, concretely: for each circuit sub-block (gate branch, choke inductor, drain branch,
thermal relaxation), go from the SPICE-level nonlinear electrothermal equations to a
reduced ODE/map that reproduces the same qualitative (and where possible quantitative)
behavior, validated against real SPICE traces, then combine the blocks into a single
reduced nTron "neuron" and finally a 2-input transponder, which serves as the single "neuron" 
for the model.

## Repo layout
- `src/transponder/` — the actual equation implementations. This is the ONLY place
  equations should live once they're validated. Import from here in notebooks; never
  redefine an equation inline in a notebook once it has graduated.
- `notebooks/` — one notebook per exploration step (one equation / one plot / one
  component at a time). Paired with `.py` files via jupytext (percent format) so git
  diffs are readable. Notebooks call into `src/transponder`, they don't define new physics
  that doesn't also get promoted to `src/`.
- `tests/` — pytest tests that check each `src/transponder` module against a SPICE
  reference trace in `data/spice_reference/` (peak values, threshold crossings, timing),
  not just "does it run."
- `data/spice_reference/` — exported ground-truth traces from actual LTspice runs
  (`SNN_2input` netlist and successors). Treat these as read-only fixtures.
- `docs/equations.md` — one section per component: the governing equation, where it came
  from (grant reference numbers / paper), what's been simplified relative to full SPICE,
  and why.