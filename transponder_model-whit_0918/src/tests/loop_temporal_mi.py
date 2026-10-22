"""Temporal proximity to criticality for ONE recurrent loop (default t4-t5-t6), after Fosque et al. 2025.

Protocol
  * the t1-t2-t3 loop is silenced (input bias 0, so it never clicks and the t3 -> t4 link carries nothing);
  * every --period ns the loop gets a kick of random size dX in {0, 1, 2}: nothing / i1 -> t5 / i1 -> t5 and i2 -> t6,
    on top of spontaneous activity from U3 switching noise (--sw_noise);
  * one long run per seed replaces many short trials, so the small network is compensated by long time.
Measure (IDTxl's JIDT discrete MI estimator, significance from its analytic surrogate distribution)
    I(tau) = I(dX ; x(t_kick + tau)),   x = number of loop spikes in a 3 ns bin, tau = 0 .. --lags bins
  Near criticality the loop remembers the kick longer (critical slowing down): I(tau) stays significant to larger tau.
  Far from it, I(tau) falls to the surrogate floor within a few bins.
Control parameter: input bias Ib1 of the loop nodes only (a per-node bias on chip).
Example:  python loop_temporal_mi.py --values 11 11.5 12 12.5 13 13.5 --runs 4 --tmax 20000
"""
import argparse, json, math, os, subprocess, sys
from multiprocessing import Pool
import numpy as np
np.math = math  # IDTxl still calls np.math, which numpy 2 removed
from idtxl.estimators_jidt import JidtDiscreteMI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--values', type=float, nargs='+', default=[11, 11.5, 12, 12.5, 13, 13.5], help='Ib1 of the loop nodes (uA)')
p.add_argument('--loop', type=int, nargs='+', default=[5, 6, 7], help='rows of the loop nodes (5,6,7 = t4,t5,t6)')
p.add_argument('--silence', type=int, nargs='+', default=[2, 3, 4], help='rows given Ib1 = 0 (2,3,4 = t1,t2,t3)')
p.add_argument('--runs', type=int, default=4, help='independent long runs (seeds) per value')
p.add_argument('--tmax', type=float, default=20000, help='ns per run')
p.add_argument('--period', type=float, default=200, help='ns between kicks (must exceed the response)')
p.add_argument('--bin', type=float, default=3.0, help='bin width in ns (= ref_period)')
p.add_argument('--lags', type=int, default=50, help='response window in bins after each kick')
p.add_argument('--fixed', default='--Ib1init 12 --Ib2init 19 --I1sw 14 --split 0.8 --sw_noise 0.3 --max_step 5e-3')
p.add_argument('--out', default='../results/loop_mi')
p.add_argument('--workers', type=int, default=os.cpu_count())
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)


def simulate(job):  # one long run; cached, so an interrupted sweep resumes
    v, seed = job
    f = f'{a.out}/loopIb1_{v:g}_seed{seed}.npz'
    if not os.path.exists(f):
        nodes = [x for n in a.silence for x in ('--ib1', str(n), '0')] + [x for n in a.loop for x in ('--ib1', str(n), str(v))]
        cmd = [sys.executable, 'toy_example.py', *a.fixed.split(), *nodes, '--pulse_period', str(a.period),
               '--tmax', str(a.tmax), '--seed', str(seed), '--save_spikes', f, '--no_plots']
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return f


def load(f):  # kick sizes (kicks,), loop activity after each kick (kicks, lags), loop branching counts
    d = np.load(f, allow_pickle=True)
    sp = [np.asarray(s, float) * 1e9 for s in d['spikes']]
    x = sum(np.histogram(sp[n], np.arange(0, a.tmax + a.bin, a.bin))[0] for n in a.loop)
    on = (d['pulse_times'] * 1e9 // a.bin).astype(int)
    ok = on + a.lags <= len(x)
    R = np.array([x[o:o + a.lags] for o in on[ok]])
    D = d['delay'] * 1e9  # descendants per loop spike, inside the loop only (as in mi_criticality_sweep.py)
    kids = [sum(np.any((sp[i] > t + D[j, i] + 1) & (sp[i] < t + D[j, i] + 8)) for i in a.loop if D[j, i] > 0)
            for j in a.loop for t in sp[j]]
    return d['pulse_sizes'][ok], R, kids


est = JidtDiscreteMI({'discretise_method': 'none', 'alph1': 3, 'alph2': 4})
rows = []
with Pool(a.workers) as pool:
    for v in a.values:
        parts = [load(f) for f in pool.map(simulate, [(v, s) for s in range(a.runs)])]
        S = np.concatenate([q[0] for q in parts])
        R = np.minimum(3, np.concatenate([q[1] for q in parts]))  # at most 3 loop spikes per bin
        I, I95 = [], []
        for k in range(a.lags):
            I.append(est.estimate(S, R[:, k]))
            I95.append(np.percentile(est.estimate_surrogates_analytic(n_perm=200, var1=S, var2=R[:, k]), 95))
        I, I95 = np.array(I), np.array(I95)
        sig = I > I95
        row = dict(value=v, kicks=int(len(S)), m_loop=float(np.mean(sum((q[2] for q in parts), []) or [np.nan])),
                   spont_per_bin=float(R[S == 0].mean()),  # loop activity with no kick = ongoing fluctuations
                   I_tau=I.tolist(), I95_tau=I95.tolist(),
                   evoked_dX2_minus_dX0=(R[S == 2].mean(0) - R[S == 0].mean(0)).tolist(),
                   integrated_MI_bits_ns=float(np.sum(np.where(sig, I, 0)) * a.bin),
                   memory_ns=float((np.nonzero(sig)[0].max() + 1) * a.bin if sig.any() else 0))
        rows.append(row)
        print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items() if not isinstance(x, list)}, flush=True)
        json.dump(rows, open(f'{a.out}/summary_loop.json', 'w'), indent=1)

tau = np.arange(a.lags) * a.bin
fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
for r in rows:
    ax[0].plot(tau, r['evoked_dX2_minus_dX0'], label=f"{r['value']:g}")
    ax[1].semilogy(tau, np.maximum(r['I_tau'], 1e-5), label=f"{r['value']:g}")
ax[0].set(title='relaxation after a kick\n(mean loop spikes/bin, dX=2 minus dX=0)', xlabel='ns after kick')
ax[1].set(title='I(dX ; loop activity at lag)', xlabel='ns after kick', ylabel='bits'); ax[1].legend(title='loop Ib1', fontsize=7)
v = [r['value'] for r in rows]
ax[2].plot(v, [r['integrated_MI_bits_ns'] for r in rows], 'o-', label='integrated significant MI [bit ns]')
ax2 = ax[2].twinx(); ax2.plot(v, [r['memory_ns'] for r in rows], 's--', color='C1'); ax2.set_ylabel('memory: last significant lag [ns]', color='C1')
ax[2].set(title='kick memory', xlabel='loop Ib1 (uA)'); ax[2].legend(fontsize=7)
ax[3].plot(v, [r['m_loop'] for r in rows], 'o-'); ax[3].axhline(1, color='r', ls=':')
ax[3].set(title='branching m inside the loop', xlabel='loop Ib1 (uA)')
fig.tight_layout(); fig.savefig(f'{a.out}/loop_temporal_mi.png', dpi=150)
