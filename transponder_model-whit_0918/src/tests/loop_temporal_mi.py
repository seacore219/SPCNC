"""Temporal proximity to criticality for ONE recurrent loop (default t4-t5-t6), after Fosque et al. 2025.

Protocol
  * the t1-t2-t3 loop is silenced (input bias 0) and the sensors are off, so the t4-t5-t6 loop is the only system;
  * the six loop biases (Ib1 and Ib2 of t4, t5, t6) are treated as weights: each sample draws them independently
    within +/-20% of the SPICE bias/switching ratios (bias3 150 uA, bias4 160 uA, channel Ic ~190 uA);
  * ongoing switching noise (--noise_dt) gives spontaneous firing; every --period ns a kick forces dX in {0..3}
    random loop nodes to spike. One long run per seed replaces many short trials.
Measures (IDTxl JIDT discrete estimators; significance = 95th percentile of their analytic surrogate distribution)
    I(tau)   = I(dX ; x(t_kick + tau))                    x = loop spikes in a 3 ns bin (0..3)
    CMI(tau) = I(dX ; x(t_kick + tau) | x_pre)            x_pre = loop spikes in the 9 ns before the kick
  memory   = last lag with significant CMI (critical slowing down: longest near criticality)
  ignition = fraction of kicks after which the loop is still active 30 ns before the next kick (0 sub -> 1 super)
  m_loop   = descendants per loop spike inside the loop (1 = critical)
Rerunning with the same arguments only simulates what is missing, then analyses everything.
Example:  python loop_temporal_mi.py --samples 32 --runs 4 --tmax 10000
"""
import argparse, json, math, os, subprocess, sys
from multiprocessing import get_context  # "fork": macOS defaults to "spawn", which re-runs this script in every worker
import numpy as np
np.math = math  # IDTxl still calls np.math, which numpy 2 removed
from idtxl.estimators_jidt import JidtDiscreteMI, JidtDiscreteCMI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--samples', type=int, default=32, help='random 6-bias settings; sample 0 is the SPICE ratio for all six')
p.add_argument('--runs', type=int, default=4, help='independent long runs (seeds) per sample')
p.add_argument('--tmax', type=float, default=10000, help='ns per run')
p.add_argument('--period', type=float, default=200, help='ns between kicks')
p.add_argument('--bin', type=float, default=3.0, help='bin width in ns (= ref_period)')
p.add_argument('--lags', type=int, default=50, help='lags (bins) after each kick used for MI')
p.add_argument('--loop', type=int, nargs='+', default=[5, 6, 7], help='rows of the loop nodes (t4, t5, t6)')
p.add_argument('--silence', type=int, nargs='+', default=[2, 3, 4], help='rows given Ib1 = 0 (t1, t2, t3)')
# toy thresholds I1sw = 14, I2sw = 20; SPICE ratios 150/190 = 0.79 and 160/190 = 0.84, +/-20%; Ib2 capped below I2sw
p.add_argument('--ib1_range', type=float, nargs=2, default=[0.79 * 0.8 * 14, 0.79 * 1.2 * 14])
p.add_argument('--ib2_range', type=float, nargs=2, default=[0.84 * 0.8 * 20, 19.9])
p.add_argument('--noise_range', type=float, nargs=2, default=[0.2, 0.8], help='switching-noise std (uA); SPICE has no noise, so it is sampled too')
p.add_argument('--fixed', default='--Ib1init 12 --Ib2init 19 --I1sw 14 --I2sw 20 --split 0.24 --noise_dt 3 --max_step 5e-3',
               help='split 0.24: SPICE gate pulse peak / driving bias = 18.5/130 uA = 0.14 = 0.606 * split')
p.add_argument('--seed', type=int, default=0, help='seed for drawing the bias samples')
p.add_argument('--splits', type=float, nargs='+', default=None, help='split sweep instead of random biases: one sample per value')
p.add_argument('--fixed_bias', type=float, nargs=3, default=[13, 19, 0.3], help='Ib1, Ib2 (all loop nodes) and sw_noise used with --splits')
p.add_argument('--out', default='../results/loop_mi')
p.add_argument('--workers', type=int, default=os.cpu_count())
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

rng = np.random.default_rng(a.seed)
B = np.column_stack([rng.uniform(*a.ib1_range, (a.samples, 3)), rng.uniform(*a.ib2_range, (a.samples, 3)),
                     rng.uniform(*a.noise_range, a.samples)])
B[0] = [0.79 * 14] * 3 + [0.84 * 20] * 3 + [np.mean(a.noise_range)]  # Ib1 of t4,t5,t6 | Ib2 of t4,t5,t6 (uA) | sw_noise
if a.splits:  # split sweep: same biases for every sample, split varies
    a.samples = len(a.splits)
    B = np.tile([a.fixed_bias[0]] * 3 + [a.fixed_bias[1]] * 3 + [a.fixed_bias[2]], (a.samples, 1))
W = int(a.period / a.bin)  # bins between kicks


def simulate(job):  # one long run; cached, so an interrupted sweep resumes
    s, seed = job
    f = f'{a.out}/sample{s}_seed{seed}.npz'
    if not os.path.exists(f):
        nodes = [x for n in a.silence for x in ('--ib1', str(n), '0')]
        nodes += [x for k, n in enumerate(a.loop) for x in ('--ib1', str(n), f'{B[s, k]:.3f}', '--ib2', str(n), f'{B[s, 3 + k]:.3f}')]
        cmd = [sys.executable, 'toy_example.py', *a.fixed.split(), *nodes, '--kick_nodes', *map(str, a.loop),
               '--sw_noise', f'{B[s, 6]:.3f}', *(['--split', str(a.splits[s])] if a.splits else []), '--pulse_period', str(a.period), '--tmax', str(a.tmax), '--seed', str(seed), '--save_spikes', f, '--no_plots']
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return f


def load(f):  # kick sizes, loop activity before / after each kick, in-loop descendants per spike
    d = np.load(f, allow_pickle=True)
    sp = [np.asarray(s, float) * 1e9 for s in d['spikes']]
    x = np.minimum(3, sum(np.histogram(sp[n], np.arange(0, a.tmax + a.bin, a.bin))[0] for n in a.loop))
    on = np.floor(d['pulse_times'] * 1e9 / a.bin + 1e-6).astype(int)  # +1e-6: 600e-9*1e9 = 599.99999...
    ok = (on >= 4) & (on + W <= len(x))
    after = np.array([x[o:o + W] for o in on[ok]]).reshape(-1, W)
    pre = np.array([min(3, x[o - 4:o - 1].sum()) for o in on[ok]], dtype=int)  # skips the bin right before the kick
    D = d['delay'] * 1e9
    kids = [sum(np.any((sp[i] > t + D[j, i] + 1) & (sp[i] < t + D[j, i] + 8)) for i in a.loop if D[j, i] > 0)
            for j in a.loop for t in sp[j]]
    return d['pulse_sizes'][ok], after, pre, kids


def curve(est, S, R, C=None):  # estimate and surrogate-95% at every lag
    out, thr = [], []
    for k in range(a.lags):
        data = dict(var1=S, var2=R[:, k]) if C is None else dict(var1=S, var2=R[:, k], conditional=C)
        out.append(est.estimate(**data))
        thr.append(np.percentile(est.estimate_surrogates_analytic(n_perm=200, **data), 95))
    return np.array(out), np.array(thr)


jobs = [(s, r) for s in range(a.samples) for r in range(a.runs)]
with get_context("fork").Pool(a.workers) as pool:  # all runs in parallel; analysis only after every run exists
    for _ in pool.imap_unordered(simulate, jobs):
        pass

mi = JidtDiscreteMI({'discretise_method': 'none', 'alph1': 4, 'alph2': 4})
cmi = JidtDiscreteCMI({'discretise_method': 'none', 'alph1': 4, 'alph2': 4, 'alphc': 4})
rows = []
for s in range(a.samples):
    parts = [load(f'{a.out}/sample{s}_seed{r}.npz') for r in range(a.runs)]
    S, A, P = (np.concatenate([q[i] for q in parts]) for i in range(3))
    kids = sum((q[3] for q in parts), [])
    I, I95 = curve(mi, S, A)
    C, C95 = curve(cmi, S, A, P)
    sig = C > C95
    sig[0] = False  # lag 0 only contains the forced kick spikes themselves
    late = A[:, -10:].sum(1) > 0  # loop still active 30 ns before the next kick
    row = dict(sample=s, split=a.splits[s] if a.splits else None, ib1=B[s, :3].round(3).tolist(), ib2=B[s, 3:6].round(3).tolist(), sw_noise=round(float(B[s, 6]), 3), kicks=int(len(S)),
               m_loop=float(np.mean(kids)) if kids else float('nan'),
               ongoing_per_bin=float(P.mean() / 3),  # loop activity just before kicks (spontaneous + leftover)
               evoked_dX3_first5=float(A[S == 3, :5].mean()) if (S == 3).any() else float('nan'),
               ignition=float(late[S > 0].mean() - late[S == 0].mean()),  # kick-driven: spontaneous firing cancels
               memory_ns=float((np.nonzero(sig)[0].max() + 1) * a.bin if sig.any() else 0),
               integrated_CMI_bits_ns=float(np.sum(np.where(sig, C, 0)) * a.bin),
               MI_tau=I.tolist(), MI95_tau=I95.tolist(), CMI_tau=C.tolist(), CMI95_tau=C95.tolist())
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if 'tau' not in k}, flush=True)
json.dump(rows, open(f'{a.out}/summary_loop.json', 'w'), indent=1)

m = np.array([r['m_loop'] for r in rows]); ong = np.array([r['ongoing_per_bin'] for r in rows])
fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
for k, key, lab in [(0, 'memory_ns', 'kick memory: last significant CMI lag [ns]'),
                    (1, 'integrated_CMI_bits_ns', 'integrated significant CMI [bit ns]'), (2, 'ignition', 'kick-driven ignition (kick - no kick)')]:
    sc = ax[k].scatter(m, [r[key] for r in rows], c=ong, cmap='viridis')
    ax[k].axvline(1, color='r', ls=':'); ax[k].set(xlabel='m inside the loop', title=lab)
fig.colorbar(sc, ax=ax[2], label='ongoing activity / node / bin')
tau = np.arange(a.lags) * a.bin
for i in np.argsort(-np.nan_to_num(m))[:5]:  # the five highest-m samples, significant lags only
    c = np.array(rows[i]['CMI_tau']); c[np.array(rows[i]['CMI95_tau']) >= c] = np.nan
    ax[3].semilogy(tau[1:], c[1:], 'o-', ms=3, label=f"m={m[i]:.2f}" + (f", split={rows[i]['split']}" if a.splits else ''))
ax[3].set(title='significant CMI(dX ; x(tau) | x_pre)', xlabel='ns after kick', ylabel='bits'); ax[3].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{a.out}/loop_temporal_mi.png', dpi=150)
