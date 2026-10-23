"""Spontaneous activity of ONE recurrent loop (default t4-t5-t6) with no kicks: only switching noise drives it.

Long runs, then simple, hardware-measurable statistics of the loop's spike train vs a swept split:
  rate          loop spikes per ns per node
  active        order parameter: fraction of 3 ns bins with at least one loop spike (~0 subcritical, ~1 self-sustained)
  bout_ns       mean duration of active bouts (spikes separated by < --gap ns), grows fastest near the transition
  fano          variance / mean of the loop spike count in --win ns windows (fluctuations; peaks near the transition)
  tau_ac_ns     autocorrelation time of binned loop activity (critical slowing down; Cramer et al. 2020)
Example:  python loop_spontaneous.py --splits 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --runs 2 --tmax 20000
"""
import argparse, json, os, subprocess, sys
from multiprocessing import get_context  # "fork": macOS defaults to "spawn", which re-runs this script in every worker
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--splits', type=float, nargs='+', required=True)
p.add_argument('--runs', type=int, default=2)
p.add_argument('--tmax', type=float, default=20000, help='ns per run')
p.add_argument('--burn', type=float, default=500, help='ns discarded at the start')
p.add_argument('--bin', type=float, default=3.0)
p.add_argument('--gap', type=float, default=20.0, help='ns of silence that ends an active bout')
p.add_argument('--win', type=float, default=30.0, help='window (ns) for the Fano factor')
p.add_argument('--loop', type=int, nargs='+', default=[5, 6, 7])
p.add_argument('--silence', type=int, nargs='+', default=[2, 3, 4])
p.add_argument('--fixed_bias', type=float, nargs=3, default=[13, 19, 0.3], help='Ib1, Ib2 (loop nodes) and sw_noise')
p.add_argument('--fixed', default='--Ib1init 12 --Ib2init 19 --I1sw 14 --I2sw 20 --noise_dt 3 --max_step 5e-3 --dt 1e9')
p.add_argument('--out', default='../results/loop_spont')
p.add_argument('--workers', type=int, default=os.cpu_count())
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)


def simulate(job):  # cached per (split value, seed), so reruns only simulate what is missing
    sp, seed = job
    f = f'{a.out}/split{sp:g}_seed{seed}.npz'
    if not os.path.exists(f):
        ib1, ib2, noise = a.fixed_bias
        nodes = [x for n in a.silence for x in ('--ib1', str(n), '0')]
        nodes += [x for n in a.loop for x in ('--ib1', str(n), str(ib1), '--ib2', str(n), str(ib2))]
        cmd = [sys.executable, 'toy_example.py', *a.fixed.split(), *nodes, '--sw_noise', str(noise), '--split', str(sp),
               '--tmax', str(a.tmax), '--seed', str(seed), '--save_spikes', f, '--no_plots']
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return f


def stats(f):
    d = np.load(f, allow_pickle=True)
    t = np.sort(np.concatenate([np.asarray(d['spikes'][n], float) * 1e9 for n in a.loop]))
    t = t[t > a.burn]; T = a.tmax - a.burn
    x = np.histogram(t, np.arange(a.burn, a.tmax + a.bin, a.bin))[0]
    w = np.histogram(t, np.arange(a.burn, a.tmax + a.win, a.win))[0]
    bouts = []
    if len(t):
        cut = np.nonzero(np.diff(t) > a.gap)[0]
        starts, ends = np.r_[t[0], t[cut + 1]], np.r_[t[cut], t[-1]]
        bouts = (ends - starts + a.bin).tolist()
    xc = x - x.mean(); ac = np.correlate(xc, xc, 'full')[len(xc) - 1:] / (xc @ xc) if xc.any() else np.zeros(1)
    k = np.argmax(ac < 1 / np.e) if (ac < 1 / np.e).any() else len(ac)
    return dict(rate=len(t) / T / len(a.loop), active=float(np.mean(x > 0)), bouts=bouts,
                fano=float(w.var() / w.mean()) if w.mean() > 0 else np.nan, tau_ac_ns=float(k * a.bin))


jobs = [(sp, r) for sp in a.splits for r in range(a.runs)]
with get_context('fork').Pool(a.workers) as pool:
    for _ in pool.imap_unordered(simulate, jobs):
        pass

rows = []
for sp in a.splits:
    S = [stats(f'{a.out}/split{sp:g}_seed{r}.npz') for r in range(a.runs)]
    b = sum((s['bouts'] for s in S), [])
    rows.append(dict(split=sp, rate=float(np.mean([s['rate'] for s in S])), active=float(np.mean([s['active'] for s in S])),
                     bout_ns=float(np.mean(b)) if b else 0.0, n_bouts=len(b), fano=float(np.nanmean([s['fano'] for s in S])),
                     tau_ac_ns=float(np.mean([s['tau_ac_ns'] for s in S]))))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
json.dump(rows, open(f'{a.out}/summary_spont.json', 'w'), indent=1)

x = [r['split'] for r in rows]
fig, ax = plt.subplots(4, 1, figsize=(6, 9), sharex=True)
for k, (key, lab) in enumerate([('active', 'order parameter\n[fraction of time active]'), ('rate', 'rate\n[spikes / ns / node]'),
                                ('bout_ns', 'mean active bout\n[ns]'), ('fano', 'Fano factor\n(fluctuations)')]):
    ax[k].plot(x, [r[key] for r in rows], 'o-'); ax[k].set_ylabel(lab)
ax[0].set_title('loop with no kicks, driven only by switching noise'); ax[-1].set_xlabel('split')
fig.tight_layout(); fig.savefig(f'{a.out}/loop_spontaneous.png', dpi=150)
