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
               '--sw_noise', f'{B[s, 6]:.3f}', *(['--split', str(a.splits[s])] if a.splits else []), '--pulse_period', str(a.period), '--tmax', str(a.tmax), '--seed', str(seed), '--save_spikes', f, '--no_plots', '--record_parents']
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
    late = [np.mean(x[o - 20:o] > 0) for o in on[on >= 20]]  # order parameter: share of active 3 ns bins in the 60 ns before a kick
    D = d['delay'] * 1e9

    def hits(j, t):  # partners of j that spike 1-8 ns after the pulse j sent at time t arrives (window shifts with the delay)
        return sum(np.any((sp[i] > t + D[j, i] + 1) & (sp[i] < t + D[j, i] + 8)) for i in a.loop if D[j, i] > 0)
    kids = [hits(j, t) for j in a.loop for t in sp[j]]
    # interventional m: single-node kicks (the forced spike sits at the kick time) minus the same count after empty kicks
    pt, ps = d['pulse_times'] * 1e9, d['pulse_sizes']
    one = [hits(j, t) for t, k in zip(pt, ps) if k == 1 for j in a.loop if np.any(np.abs(sp[j] - t) < 0.05)]
    none = [hits(j, t) for t, k in zip(pt, ps) if k == 0 for j in a.loop]
    # exact genealogy (runs made with --record_parents): each child counts 1, shared equally among its parents
    credit, gen, sizes = [], [], []
    if 'parents' in d:
        spikes = sorted((t, n, k) for n in a.loop for k, t in enumerate(sp[n]))  # time order: parents come first
        pos = {(n, k): i for i, (t, n, k) in enumerate(spikes)}
        cr, g, root = np.zeros(len(spikes)), np.zeros(len(spikes), int), list(range(len(spikes)))
        def find(i):  # avalanche label (union-find, iterative so long cascades are fine)
            while root[i] != i:
                root[i] = root[root[i]]; i = root[i]
            return i
        for i, (t, n, k) in enumerate(spikes):
            ps = [pos[j, int(np.argmin(np.abs(sp[j] - tc * 1e9)))] for j, tc in d['parents'][n][k] if j in a.loop]
            for q in ps:
                cr[q] += 1 / len(ps)  # 1 / number of parents, whatever that number is
                root[find(q)] = find(i)  # child and parents belong to the same avalanche
            g[i] = 1 + max(g[q] for q in ps) if ps else 0
        credit, gen = cr.tolist(), g.tolist()
        sizes = np.bincount([find(i) for i in range(len(spikes))]).tolist() if spikes else []
        sizes = [x for x in sizes if x > 0]
    return d['pulse_sizes'][ok], after, pre, kids, one, none, credit, gen, sizes, late


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

from idtxl.estimators_jidt import JidtDiscreteMI, JidtDiscreteCMI  # loaded only now, so simulations never depend on IDTxl
mi = JidtDiscreteMI({'discretise_method': 'none', 'alph1': 4, 'alph2': 4})
cmi = JidtDiscreteCMI({'discretise_method': 'none', 'alph1': 4, 'alph2': 4, 'alphc': 4})
rows = []
for s in range(a.samples):
    parts = [load(f'{a.out}/sample{s}_seed{r}.npz') for r in range(a.runs)]
    S, A, P = (np.concatenate([q[i] for q in parts]) for i in range(3))
    kids = sum((q[3] for q in parts), [])
    one, none = sum((q[4] for q in parts), []), sum((q[5] for q in parts), [])
    credit, gen, sizes = (np.array(sum((q[i] for q in parts), [])) for i in (6, 7, 8))
    order = float(np.mean(sum((q[9] for q in parts), [])))
    I, I95 = curve(mi, S, A)
    C, C95 = curve(cmi, S, A, P)
    sig = C > C95
    sig[0] = False  # lag 0 only contains the forced kick spikes themselves
    sigI = I > I95; sigI[0] = False  # plain MI: lower chance threshold; best readout when ongoing activity is low
    late = A[:, -10:].sum(1) > 0  # loop still active 30 ns before the next kick
    row = dict(sample=s, split=a.splits[s] if a.splits else None, ib1=B[s, :3].round(3).tolist(), ib2=B[s, 3:6].round(3).tolist(), sw_noise=round(float(B[s, 6]), 3), kicks=int(len(S)),
               m_loop=float(np.mean(kids)) if kids else float('nan'),
               order_param=order,  # fraction of 3 ns bins with a loop spike, 140-200 ns after the previous kick
               m_exact=float(credit.mean()) if credit.size else float('nan'),  # children per spike from the recorded causes
               m_by_generation=[float(credit[gen == k].mean()) for k in range(6) if (gen == k).any()],
               avalanche_mean=float(sizes.mean()) if sizes.size else float('nan'), avalanche_max=int(sizes.max()) if sizes.size else 0,
               m_causal=float(np.mean(one) - np.mean(none)) if one and none else float('nan'), single_kicks=len(one),
               ongoing_per_bin=float(P.mean() / 3),  # loop activity just before kicks (spontaneous + leftover)
               evoked_dX3_first5=float(A[S == 3, :5].mean()) if (S == 3).any() else float('nan'),
               ignition=float(late[S > 0].mean() - late[S == 0].mean()),  # kick-driven: spontaneous firing cancels
               memory_ns=float((np.nonzero(sig)[0].max() + 1) * a.bin if sig.any() else 0),
               integrated_CMI_bits_ns=float(np.sum(np.where(sig, C, 0)) * a.bin),
               memory_MI_ns=float(np.nonzero(sigI)[0].max() * a.bin if sigI.any() else 0),  # lag (start of bin) of last significant MI
               integrated_MI_bits_ns=float(np.sum(np.where(sigI, I, 0)) * a.bin),
               MI_tau=I.tolist(), MI95_tau=I95.tolist(), CMI_tau=C.tolist(), CMI95_tau=C95.tolist())
    rows.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if 'tau' not in k}, flush=True)
json.dump(rows, open(f'{a.out}/summary_loop.json', 'w'), indent=1)

# figure: left = MI vs lag after the kick (significant lags, one colour per setting); right = br and MI vs split, shared x axis
x = np.array([r['split'] for r in rows]) if a.splits else np.arange(len(rows))
xlab = 'split' if a.splits else 'sample'
fig = plt.figure(figsize=(17, 8.5))
gs = fig.add_gridspec(3, 2, width_ratios=[1.9, 1], hspace=0.1, wspace=0.22)
axT, axB, axI, axK = (fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]),
                      fig.add_subplot(gs[2, 1]))
tau = np.arange(a.lags) * a.bin
cmap, norm = plt.cm.viridis, plt.Normalize(x.min(), x.max() if x.max() > x.min() else x.min() + 1)
for r, xv in zip(rows, x):
    v = np.array(r['MI_tau']); keep = (v > np.array(r['MI95_tau'])) & (tau > 0)  # significant lags only, lag 0 left out
    axT.scatter(tau[keep], v[keep], s=14, color=cmap(norm(xv)))
axT.set(yscale='log', xlabel='ns after the kick', ylabel='I(kick size ; loop spikes at that lag)  [bits]',
        title='mutual information vs time after the kick (significant lags only)')
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axT, label=xlab)
axB.plot(x, [r['m_exact'] for r in rows], 'o-', color='k')
axB.axhline(1, color='r', ls=':'); axB.set(ylabel='branching ratio br', title='branching ratio and mutual information vs ' + xlab)
axB.tick_params(labelbottom=False)
axI.plot(x, [r['integrated_MI_bits_ns'] for r in rows], 'o-', color='tab:orange')
axI.set(ylabel='integrated MI  [bit ns]'); axI.tick_params(labelbottom=False)
axK.plot(x, [r.get('order_param', np.nan) for r in rows], 'o-', color='tab:green')
axK.set(xlabel=xlab, ylabel='order parameter\n[fraction of time active]', ylim=(-0.03, 1.03))
axK.text(0.02, 0.85, 'loop active 140-200 ns after a kick (self-sustained)', transform=axK.transAxes, fontsize=8, color='0.3')
axI.sharex(axB); axK.sharex(axB)
fig.savefig(f'{a.out}/loop_temporal_mi.png', dpi=150, bbox_inches='tight')
