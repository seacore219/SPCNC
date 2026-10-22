"""Mutual information (IDTxl) as a proximity-to-criticality measure for the toy transponder network.

For each value of one control parameter (default: the input bias Ib1, the knob you can turn on a real chip):
  1. run toy_example.py `--trials` times with Poisson sensor input and U3 switching noise (different seeds),
  2. bin every node's spike train at bin = refractory period (3 ns), as in Cramer et al. 2020,
  3. IDTxl BivariateMI network inference with the discrete JIDT estimator:
       I_in  = MI between the sensors' past and each transponder (stimulus -> network, Shew et al. 2011)
       I_int = MI between the other transponders' past and each transponder (pairwise/lagged I, Barnett 2013, Cramer 2020)
     only statistically significant MI (IDTxl surrogate tests) is counted,
  4. an independent criticality check: branching parameter m (descendants per spike, using the known wiring),
     m -> 1 at criticality.
Example:  python mi_criticality_sweep.py --values 10 11 12 13 --trials 20
"""
import argparse, json, math, os, subprocess, sys
from multiprocessing import Pool
import numpy as np
np.math = math  # IDTxl still calls np.math, which numpy 2 removed
from idtxl.data import Data
from idtxl.bivariate_mi import BivariateMI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--param', default='Ib1init', help='toy_example.py flag to sweep (e.g. Ib1init, split, Ib2init)')
p.add_argument('--values', type=float, nargs='+', default=[10, 10.5, 11, 11.5, 12, 12.5, 13])
p.add_argument('--trials', type=int, default=20, help='independent runs per value = IDTxl replications')
p.add_argument('--tmax', type=float, default=300, help='ns per run')
p.add_argument('--bin', type=float, default=3.0, help='bin width in ns (= ref_period)')
p.add_argument('--max_lag', type=int, default=5, help='largest source lag in bins tested by IDTxl')
p.add_argument('--fixed', default='--Ib1init 12 --Ib2init 19 --I1sw 14 --split 0.5 --rate 0.033 --sw_noise 0.3 --max_step 5e-3',
               help='toy_example.py flags held fixed (the swept flag is appended after them and wins)')
p.add_argument('--out', default='../results/mi_sweep')
p.add_argument('--workers', type=int, default=os.cpu_count())
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
SENS, TR = [0, 1], list(range(2, 8))  # node rows: i1, i2 | t1..t6


def simulate(job):  # one toy_example.py run; skipped if its file already exists, so a sweep can be resumed
    v, seed = job
    f = f'{a.out}/{a.param}_{v:g}_seed{seed}.npz'
    if not os.path.exists(f):
        cmd = [sys.executable, 'toy_example.py', *a.fixed.split(), f'--{a.param}', str(v), '--tmax', str(a.tmax),
               '--seed', str(seed), '--save_spikes', f, '--no_plots']
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return f


def binned(f):  # (8 nodes, n_bins) array of 0/1
    sp = np.load(f, allow_pickle=True)['spikes']
    edges = np.arange(0, a.tmax + a.bin, a.bin)
    return np.array([np.minimum(1, np.histogram(np.asarray(s, float) * 1e9, edges)[0]) for s in sp])


def branching(files, lo=1.0, hi=8.0):
    """m = mean number of downstream transponders that spike lo..hi ns after a transponder's pulse arrives
    (arrival = spike + line delay; click-to-spike latency is ~4.75 ns). m < 1 activity dies out, m > 1 it grows."""
    kids = []
    for f in files:
        sp = [np.asarray(s, float) * 1e9 for s in np.load(f, allow_pickle=True)['spikes']]
        for j in TR:
            for t in sp[j]:
                kids.append(sum(np.any((sp[i] > t + D[j, i] + lo) & (sp[i] < t + D[j, i] + hi)) for i in TR if D[j, i] > 0))
    return float(np.mean(kids)) if kids else np.nan


settings = {'cmi_estimator': 'JidtDiscreteCMI', 'discretise_method': 'none', 'n_discrete_bins': 2,
            'alph1': 2, 'alph2': 2, 'alphc': 2, 'min_lag_sources': 1, 'max_lag_sources': a.max_lag, 'verbose': False}


def omnibus(res):  # significant MI (bits) that each transponder's present shares with its selected sources' past
    out = []
    for t in TR:
        mi = res.get_single_target(t, fdr=False)['omnibus_mi']
        out.append(0.0 if mi is None else float(mi))
    return np.array(out)


subprocess.run([sys.executable, 'toy_example.py', '--tmax', '0.01', '--save_spikes', f'{a.out}/_wiring.npz', '--no_plots'],
               check=True, stdout=subprocess.DEVNULL)
D = np.load(f'{a.out}/_wiring.npz', allow_pickle=True)['delay'] * 1e9  # wiring/delays (ns) read from toy_example.py itself

rows = []
with Pool(a.workers) as pool:
    for v in a.values:
        files = pool.map(simulate, [(v, s) for s in range(a.trials)])
        X = np.stack([binned(f) for f in files], axis=2)  # (processes, samples, replications) = IDTxl 'psr'
        data = Data(X, dim_order='psr', normalise=False)
        I_in = omnibus(BivariateMI().analyse_network(settings, data, targets=TR, sources=[SENS] * len(TR)))
        I_int = omnibus(BivariateMI().analyse_network(settings, data, targets=TR, sources=[[s for s in TR if s != t] for t in TR]))
        pr = X[TR].mean(axis=(1, 2))  # spike probability per bin, per transponder
        H = np.where((pr > 0) & (pr < 1), -pr * np.log2(np.where(pr > 0, pr, 1)) - (1 - pr) * np.log2(np.where(pr < 1, 1 - pr, 1)), 0)
        row = dict(value=v, rate_GHz=float(pr.mean() / a.bin), H=float(H.mean()), m=branching(files),
                   I_in=float(I_in.mean()), I_int=float(I_int.mean()),
                   I_in_over_H=float(np.mean(I_in / np.where(H > 0, H, np.inf))),
                   I_int_over_H=float(np.mean(I_int / np.where(H > 0, H, np.inf))),
                   I_in_per_node=I_in.tolist(), I_int_per_node=I_int.tolist())
        rows.append(row)
        print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items() if 'per_node' not in k}, flush=True)
        json.dump(rows, open(f'{a.out}/summary_{a.param}.json', 'w'), indent=1)

v = [r['value'] for r in rows]
fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
ax[0].plot(v, [r['rate_GHz'] for r in rows], 'o-'); ax[0].set(title='mean firing rate', ylabel='spikes / ns')
ax[1].plot(v, [r['m'] for r in rows], 'o-'); ax[1].axhline(1, color='r', ls=':'); ax[1].set(title='branching m (descendants / spike)')
ax[2].plot(v, [r['I_in'] for r in rows], 'o-', label='I_in (sensors -> node)')
ax[2].plot(v, [r['I_int'] for r in rows], 's-', label='I_int (other nodes -> node)'); ax[2].set(title='IDTxl MI [bits]'); ax[2].legend(fontsize=8)
ax[3].plot(v, [r['I_in_over_H'] for r in rows], 'o-', label='I_in / H')
ax[3].plot(v, [r['I_int_over_H'] for r in rows], 's-', label='I_int / H'); ax[3].set(title='MI normalised by entropy'); ax[3].legend(fontsize=8)
for x in ax: x.set_xlabel(a.param)
fig.tight_layout(); fig.savefig(f'{a.out}/mi_sweep_{a.param}.png', dpi=150)
