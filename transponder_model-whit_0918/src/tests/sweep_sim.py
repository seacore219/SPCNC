"""Simulation-only sweep driver for the t4-t5-t6 loop: runs toy_example.py and saves the spike files, no analysis.

The simulation commands and file names are copied verbatim from loop_temporal_mi.py (kick protocol) and
loop_spontaneous.py (no kicks), so every .npz is identical to what those scripts produce, and folders already
filled by them are reused (existing files are skipped). It also writes manifest.json: for every file, the exact
parameters and command that made it, so the sample number never has to be decoded by hand.

Each .npz holds: spikes (output spike times in s, one array per node: i1, i2, t1..t6), tmax, delay (s),
pulse_times (s), pulse_sizes (kick sizes); with kicks also parents (recorded causes of every spike);
with --save_currents also t and y (solver time points and state: i_nw, i_loop per node).

Modes (pick one):
  --splits 0.5 0.6 ...      kick protocol, all loop nodes at --fixed_bias, one setting per split value
  --samples N               kick protocol, N random settings of the 6 loop biases + noise (sample 0 = SPICE ratios)
  --no_kicks --splits ...   no kicks, only switching noise (loop_spontaneous.py files)
Example:  python sweep_sim.py --splits 0.5 0.6 0.7 0.8 0.9 --runs 8 --tmax 10000 --out ../results/loop_split_fine
"""
import argparse, json, os, subprocess, sys
from multiprocessing import get_context  # "fork": macOS defaults to "spawn", which re-runs this script in every worker
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--samples', type=int, default=32, help='random 6-bias settings; sample 0 is the SPICE ratio for all six')
p.add_argument('--splits', type=float, nargs='+', default=None, help='split sweep: one setting per value')
p.add_argument('--no_kicks', action='store_true', help='no kicks: noise-driven runs (needs --splits)')
p.add_argument('--runs', type=int, default=4, help='independent runs (seeds 0..runs-1) per setting')
p.add_argument('--tmax', type=float, default=10000, help='ns per run')
p.add_argument('--period', type=float, default=200, help='ns between kicks')
p.add_argument('--loop', type=int, nargs='+', default=[5, 6, 7], help='rows of the loop nodes (t4, t5, t6)')
p.add_argument('--silence', type=int, nargs='+', default=[2, 3, 4], help='rows given Ib1 = 0 (t1, t2, t3)')
p.add_argument('--ib1_range', type=float, nargs=2, default=[0.79 * 0.8 * 14, 0.79 * 1.2 * 14])
p.add_argument('--ib2_range', type=float, nargs=2, default=[0.84 * 0.8 * 20, 19.9])
p.add_argument('--noise_range', type=float, nargs=2, default=[0.2, 0.8])
p.add_argument('--fixed', default=None, help='toy_example.py flags held fixed (defaults as in the original scripts)')
p.add_argument('--fixed_bias', type=float, nargs=3, default=[13, 19, 0.3], help='Ib1, Ib2 (all loop nodes) and sw_noise')
p.add_argument('--seed', type=int, default=0, help='seed for drawing the random bias samples')
p.add_argument('--save_currents', action='store_true', help='also save solver time points and currents (much larger files)')
p.add_argument('--out', required=True)
p.add_argument('--workers', type=int, default=os.cpu_count())
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
if a.no_kicks and not a.splits:
    p.error('--no_kicks needs --splits')
if a.fixed is None:  # same defaults as loop_temporal_mi.py / loop_spontaneous.py
    a.fixed = ('--Ib1init 12 --Ib2init 19 --I1sw 14 --I2sw 20 --noise_dt 3 --max_step 5e-3 --dt 1e9' if a.no_kicks else
               '--Ib1init 12 --Ib2init 19 --I1sw 14 --I2sw 20 --split 0.24 --noise_dt 3 --max_step 5e-3')
extra = ['--save_currents'] if a.save_currents else []

# ---- settings (identical to loop_temporal_mi.py) ----
rng = np.random.default_rng(a.seed)
B = np.column_stack([rng.uniform(*a.ib1_range, (a.samples, 3)), rng.uniform(*a.ib2_range, (a.samples, 3)),
                     rng.uniform(*a.noise_range, a.samples)])
B[0] = [0.79 * 14] * 3 + [0.84 * 20] * 3 + [np.mean(a.noise_range)]  # Ib1 of t4,t5,t6 | Ib2 of t4,t5,t6 (uA) | sw_noise
if a.splits:
    a.samples = len(a.splits)
    B = np.tile([a.fixed_bias[0]] * 3 + [a.fixed_bias[1]] * 3 + [a.fixed_bias[2]], (a.samples, 1))


def job_cmd(s, seed):
    """File name and toy_example.py command for setting s, seed (verbatim from the two sweep scripts)."""
    if a.no_kicks:  # loop_spontaneous.py
        sp = a.splits[s]; ib1, ib2, noise = a.fixed_bias
        f = f'{a.out}/split{sp:g}_seed{seed}.npz'
        nodes = [x for n in a.silence for x in ('--ib1', str(n), '0')]
        nodes += [x for n in a.loop for x in ('--ib1', str(n), str(ib1), '--ib2', str(n), str(ib2))]
        cmd = [sys.executable, 'toy_example.py', *a.fixed.split(), *nodes, '--sw_noise', str(noise), '--split', str(sp),
               '--tmax', str(a.tmax), '--seed', str(seed), '--save_spikes', f, '--no_plots', *extra]
    else:  # loop_temporal_mi.py
        f = f'{a.out}/sample{s}_seed{seed}.npz'
        nodes = [x for n in a.silence for x in ('--ib1', str(n), '0')]
        nodes += [x for k, n in enumerate(a.loop) for x in ('--ib1', str(n), f'{B[s, k]:.3f}', '--ib2', str(n), f'{B[s, 3 + k]:.3f}')]
        cmd = [sys.executable, 'toy_example.py', *a.fixed.split(), *nodes, '--kick_nodes', *map(str, a.loop),
               '--sw_noise', f'{B[s, 6]:.3f}', *(['--split', str(a.splits[s])] if a.splits else []), '--pulse_period', str(a.period),
               '--tmax', str(a.tmax), '--seed', str(seed), '--save_spikes', f, '--no_plots', '--record_parents', *extra]
    return f, cmd


def simulate(job):  # skipped if the file already exists, so a sweep resumes and reuses old runs
    f, cmd = job_cmd(*job)
    if not os.path.exists(f):
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return f


jobs = [(s, r) for s in range(a.samples) for r in range(a.runs)]
manifest = []
for s, r in jobs:
    f, cmd = job_cmd(s, r)
    manifest.append(dict(file=os.path.basename(f), setting=s, seed=r, split=a.splits[s] if a.splits else None,
                         ib1=B[s, :3].round(3).tolist(), ib2=B[s, 3:6].round(3).tolist(), sw_noise=round(float(B[s, 6]), 3),
                         kicks=not a.no_kicks, tmax_ns=a.tmax, command=' '.join(cmd[1:])))
json.dump(manifest, open(f'{a.out}/manifest.json', 'w'), indent=1)  # written first, so it exists even mid-sweep
with get_context('fork').Pool(a.workers) as pool:
    for i, f in enumerate(pool.imap_unordered(simulate, jobs), 1):
        print(f'{i}/{len(jobs)} {os.path.basename(f)}', flush=True)
