#!/usr/bin/env python3
"""
Six-transponder schematic for the toy network (toy_example.py / notebook N06).

Every box is one copy of the single-transponder schematic (single_transponder_simple.pdf), drawn with
the same parts and the same four current loops. The two pulsers inside that drawing are replaced by the
links of the network: each incoming link enters U3's gate through its own delay line (the `delay`
matrix entry) and its own R_in / L_in branch.  Blue monospace tags are the variable names used in the
toy code (notebook sections 1-2).

    python3 make_network_schematic.py            -> N06_network_schematic.svg
    python3 make_network_schematic.py --render   -> also .png and .pdf (needs playwright + chromium)

Keep `delay_ns` in sync with notebook section 3.
"""
import html
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent

# ----------------------------------------------------------------------------- wiring (notebook section 3)
names = ['i1', 'i2', 't1', 't2', 't3', 't4', 't5', 't6']
#                    to: i1 i2 t1 t2 t3 t4 t5 t6    from:
delay_ns = np.array([[0, 0, 5, 0, 0, 0, 9, 0],   # i1
                     [0, 0, 0, 5, 0, 0, 0, 5],   # i2
                     [0, 0, 0, 5, 5, 0, 0, 0],   # t1
                     [0, 0, 5, 0, 5, 0, 0, 0],   # t2
                     [0, 0, 5, 5, 0, 5, 0, 0],   # t3
                     [0, 0, 0, 0, 0, 0, 5, 5],   # t4
                     [0, 0, 0, 0, 0, 5, 0, 5],   # t5
                     [0, 0, 0, 0, 0, 5, 5, 0]])  # t6

# values printed on the drawing (notebook defaults, section 1)
Ib1_uA, Ib2_uA = 12.0, 19.0
ib1_over = {2: 13.0, 4: 11.2}
ib2_over = {}
ref_ns = 3

# ----------------------------------------------------------------------------- style
INK = '#3a4a5a'
GREY = '#6b7a89'
BLACK = '#1d2733'
BLUE = '#2457c5'      # toy-code variable names
LOOP = {1: '#7b4fd6', 2: '#d6457a', 3: '#138a96', 4: '#c9821a'}
WIRE = {'i1': '#1565c0', 'i2': '#2e7d32', 't1': '#c62828', 't2': '#ef6c00',
        't3': '#6a1b9a', 't4': '#00838f', 't5': '#ad1457', 't6': '#5d4037'}
MONO = 'ui-monospace, Menlo, Consolas, monospace'


def esc(s):
    return html.escape(str(s), quote=False)


def T(x, y, s, size=14, fill=BLACK, weight=400, anchor='start', italic=False, mono=False):
    fam = f' font-family="{MONO}"' if mono else ''
    st = ' font-style="italic"' if italic else ''
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" font-weight="{weight}" '
            f'text-anchor="{anchor}"{st}{fam}>{esc(s)}</text>')


def tag(x, y, s, anchor='start', size=12, weight=400):
    return T(x, y, s, size=size, fill=BLUE, weight=weight, anchor=anchor, mono=True)


def P(d, w=2.6, fill='none', extra=''):
    return f'<path d="{d}" fill="{fill}" stroke="{INK}" stroke-width="{w}" {extra}/>'


def poly(pts, w=2.6, color=INK):
    s = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{w}" stroke-linejoin="round" points="{s}"/>'


def dot(x, y, r=5, fill=INK):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"/>'


def zig_h(x1, x2, y, amp=12, n=12):
    st = (x2 - x1) / n
    pts = [(x1, y)] + [(x1 + i * st, y + (amp if i % 2 else -amp)) for i in range(1, n)] + [(x2, y)]
    return poly(pts)


def zig_v(x, y1, y2, amp=12, n=12):
    st = (y2 - y1) / n
    pts = [(x, y1)] + [(x + (-amp if i % 2 else amp), y1 + i * st) for i in range(1, n)] + [(x, y2)]
    return poly(pts)


def ind_h(x, y, n=4, w=17.5, r=8.8):
    return P(f'M{x} {y}' + f' a{r} {r} 0 0 1 {w} 0' * n)


def ind_v(x, y, n=4, h=20, r=10):
    return P(f'M{x} {y}' + f' a{r} {r} 0 0 1 0 {h}' * n)


def ground(x, y):
    return P(f'M{x} {y}V{y + 14}') + P(f'M{x - 18} {y + 14}h36M{x - 12} {y + 21}h24M{x - 6} {y + 28}h12')


# ----------------------------------------------------------------------------- one transponder (local coords)
def slot_y(i, k):
    """y of input slot i (0 = top) when the node has k inputs; the last slot is always at the gate (y=420)."""
    return 420 - 60 * (k - 1 - i)


def module(n, srcs):
    """srcs = [(sender_name, delay_ns), ...] in slot order. Returns SVG for local coordinates."""
    k = len(srcs)
    nm = names[n]
    b1 = ib1_over.get(n, Ib1_uA)
    b2 = ib2_over.get(n, Ib2_uA)
    n_out = int((delay_ns[n] > 0).sum())
    o = []

    # loop regions (same shapes as the single-transponder drawing)
    y0 = slot_y(0, k)
    o.append(f'<path d="M110 {y0 - 30} H372 V404 H450 V502 H110 Z" fill="{LOOP[1]}" fill-opacity=".07" '
             f'stroke="{LOOP[1]}" stroke-width="3" stroke-dasharray="9 6" stroke-linejoin="round"/>')
    o.append(f'<path d="M488 198 H622 V502 H488 Z" fill="{LOOP[2]}" fill-opacity=".07" stroke="{LOOP[2]}" '
             f'stroke-width="3" stroke-dasharray="9 6" stroke-linejoin="round"/>')
    o.append(f'<path d="M658 198 H942 V402 H1122 V502 H658 Z" fill="{LOOP[3]}" fill-opacity=".07" '
             f'stroke="{LOOP[3]}" stroke-width="3" stroke-dasharray="9 6" stroke-linejoin="round"/>')
    o.append(f'<path d="M1168 198 H1362 V502 H1168 Z" fill="{LOOP[4]}" fill-opacity=".07" stroke="{LOOP[4]}" '
             f'stroke-width="3" stroke-dasharray="9 6" stroke-linejoin="round"/>')
    for c, d in ((1, 'M420 502 l-40 0'), (2, 'M540 198 l40 0'), (3, 'M760 198 l40 0'), (4, 'M1220 198 l40 0')):
        o.append(f'<path d="{d}" stroke="{LOOP[c]}" stroke-width="3" marker-end="url(#ah{c})"/>')
    o.append(T(130, 470, '① input loop', 14, LOOP[1], 700))
    o.append(T(130, 488, 'gate current into U3', 12, LOOP[1]))
    o.append(tag(250, 488, '= Σ amp·OutputPulse', size=11))
    o.append(T(500, 455, '② Q1 shunt loop', 14, LOOP[2], 700))
    o.append(T(500, 473, 'catches current', 12, LOOP[2]))
    o.append(T(500, 489, 'while U3 is resistive', 12, LOOP[2]))
    o.append(T(740, 455, '③ storage loop', 14, LOOP[3], 700))
    o.append(T(740, 473, 'I_loop circulates; drains via R2;', 12, LOOP[3]))
    o.append(T(740, 489, 'drives U2 gate', 12, LOOP[3]))
    o.append(T(1180, 455, '④ output loop', 14, LOOP[4], 700))
    o.append(T(1180, 473, 'U2 current pushed', 12, LOOP[4]))
    o.append(T(1180, 489, 'into L2 / R1 → links', 12, LOOP[4]))

    # common ground
    o.append(P('M100 520H1470'))
    o.append(ground(800, 520))
    o.append(T(822, 548, 'common ground (0 V)', 12, GREY))

    # input stage: one branch (delay line + R_in + L_in) per incoming link
    for i, (s, d) in enumerate(srcs):
        y = slot_y(i, k)
        c = WIRE[s]
        o.append(P(f'M70 {y}H84'))
        o.append(f'<rect x="84" y="{y - 13}" width="66" height="26" rx="4" fill="#f3f6fa" stroke="{INK}" stroke-width="2"/>')
        o.append(T(117, y + 5, f'{d} ns', 13, BLACK, 600, 'middle'))
        o.append(P(f'M150 {y}H160'))
        o.append(zig_h(160, 240, y))
        o.append(P(f'M240 {y}H265'))
        o.append(ind_h(265, y))
        o.append(P(f'M335 {y}H380'))
        o.append(T(72, y - 20, f'from {s}', 13, c, 700))
        o.append(T(200, y - 17, 'R_in · 20 Ω', 11, GREY, 600, 'middle'))
        o.append(T(300, y - 17, 'L_in · 5 nH', 11, GREY, 600, 'middle'))
        if y < 420 or k > 1:
            o.append(dot(380, y))
    o.append(P(f'M380 {y0}V420 M380 420H440'))
    o.append(dot(380, 420))

    # U3 (input nTron)
    o.append(f'<rect x="459" y="250" width="22" height="220" rx="3" fill="#fff" stroke="{INK}" stroke-width="2.6"/>')
    o.append(P('M459 420L440 412L440 428Z', fill=INK))
    o.append(T(488, 254, 'd', 13, GREY))
    o.append(T(488, 468, 's', 13, GREY))
    o.append(T(434, 408, 'g', 13, GREY, anchor='end'))
    o.append(T(494, 296, 'U3', 20, BLACK, 600))
    o.append(T(494, 316, 'input nTron (Q1)', 13, GREY))
    o.append(T(494, 333, '≈ L_nw1, R_nw1(t)', 12, GREY, italic=True))
    o.append(tag(494, 352, 'Inw1_sw'))
    o.append(tag(494, 368, 'Lnw'))
    o.append(tag(494, 384, 'Rnw ← Rnorm'))
    o.append(P('M470 250V180 M470 470V520'))
    o.append(dot(470, 520)); o.append(dot(470, 180))
    o.append(tag(452, 300, 'i_nw', anchor='end'))

    # bias3
    o.append(P('M470 80V180'))
    o.append(P('M470 140l-8 -14h16z', fill=INK))
    o.append(T(486, 104, 'bias3 in', 16, BLACK, 600))
    o.append(T(486, 122, 'I_b1 ≈ 150 µA (1.5 V via 10 kΩ)', 12, GREY))
    o.append(tag(486, 142, f'Ib1 = {b1:g} µA · transponder_bias1[{n}]'))

    # rail to storage loop, shunt branch (L1 + R3), R2, Lb1
    o.append(P('M470 180H760'))
    o.append(dot(640, 180))
    o.append(P('M640 180V215') + ind_v(640, 215) + P('M640 295V320') + zig_v(640, 320, 405) + P('M640 405V520'))
    o.append(dot(640, 520))
    o.append(T(658, 262, 'L1 · 5 nH', 14, BLACK, 600)); o.append(tag(658, 278, 'Ls1'))
    o.append(T(658, 370, 'R3 · 3 Ω', 14, BLACK, 600))
    o.append(T(658, 388, 'L_s1, R_s1', 12, GREY, italic=True)); o.append(tag(658, 404, 'Rs1'))
    o.append(zig_h(760, 850, 180))
    o.append(P('M850 180H960'))
    o.append(T(805, 160, 'R2 · 4 Ω', 14, BLACK, 600, 'middle'))
    o.append(T(805, 214, 'R_T(t)', 12, GREY, anchor='middle', italic=True))
    o.append(tag(805, 232, 'RT0 (+U2 when fired)', anchor='middle'))
    o.append(P('M960 180V230') + ind_v(960, 230, h=25, r=12.5) + P('M960 330V420H1120'))
    o.append(T(978, 272, 'Lb1 ≈ 148 nH', 14, BLACK, 600))
    o.append(T(978, 290, 'storage inductor (L_T)', 12, GREY, italic=True))
    o.append(tag(978, 308, 'LT'))
    o.append(tag(800, 340, 'i_loop', anchor='middle'))

    # U2 (output nTron)
    o.append(f'<rect x="1139" y="250" width="22" height="220" rx="3" fill="#fff" stroke="{INK}" stroke-width="2.6"/>')
    o.append(P('M1139 420L1120 412L1120 428Z', fill=INK))
    o.append(T(1168, 254, 'd', 13, GREY))
    o.append(T(1168, 468, 's', 13, GREY))
    o.append(T(1114, 408, 'g', 13, GREY, anchor='end'))
    o.append(T(1174, 296, 'U2', 20, BLACK, 600))
    o.append(T(1174, 316, 'output nTron (Q2)', 13, GREY))
    o.append(T(1174, 333, '≈ L_nw2, R_nw2(t)', 12, GREY, italic=True))
    o.append(tag(1174, 352, 'Inw2_sw'))
    o.append(tag(1174, 368, 'Rnorm (adds to RT)'))
    o.append(P('M1150 250V180 M1150 470V520'))
    o.append(dot(1150, 520)); o.append(dot(1150, 180))

    # bias4
    o.append(P('M1150 80V180'))
    o.append(P('M1150 140l-8 -14h16z', fill=INK))
    o.append(T(1166, 104, 'bias4 in', 16, BLACK, 600))
    o.append(T(1166, 122, 'I_b2 ≈ 160 µA (1.6 V via 10 kΩ)', 12, GREY))
    o.append(tag(1166, 142, f'Ib2 = {b2:g} µA · transponder_bias2[{n}]'))

    # output: L2, junction, R1 shunt, links leave to the right
    o.append(P('M1150 180H1200') + ind_h(1200, 180) + P('M1270 180H1640'))
    o.append(T(1235, 160, 'L2 · 5 nH', 14, BLACK, 600, 'middle'))
    o.append(T(1235, 214, 'L_out', 12, GREY, anchor='middle', italic=True))
    o.append(tag(1235, 232, 'not modelled', anchor='middle', size=11))
    o.append(dot(1380, 180))
    o.append(P('M1380 180V300') + zig_v(1380, 300, 385) + P('M1380 385V520'))
    o.append(dot(1380, 520))
    o.append(T(1398, 340, 'R1 · 3 Ω', 14, BLACK, 600))
    o.append(T(1398, 358, 'R_s2', 12, GREY, italic=True))
    o.append(tag(1398, 374, 'not modelled', size=11))
    o.append(tag(1398, 236, 'split', weight=700, size=14))
    o.append(tag(1398, 252, f'fan-out n = {n_out}', size=11))
    o.append(tag(1398, 267, f'each link: Ib2·split/{n_out}', size=11))
    o.append(T(1398, 282, 'rest (1−split) → R1', 11, GREY))
    o.append(P('M1520 180l-12 -7v14z', fill=INK, w=0))
    o.append(T(1560, 166, 'links →', 13, BLACK, 600))

    # title block
    o.append(T(70, 42, nm, 28, BLACK, 700))
    fan_in = ', '.join(s for s, _ in srcs)
    o.append(T(70, 64, f'matrix row/col {n} · inputs: {fan_in}', 13, GREY))
    o.append(tag(70, 84, f'ref_period = {ref_ns} ns', size=11))
    return '\n'.join(o)


# ----------------------------------------------------------------------------- layout
# Positions mirror the network graph in the notebook (section 3):
#   t1 (top) / t2 (bottom)  ->  t3 (middle)  ==feedforward==>  t4 (middle)  ->  t5 (top) / t6 (bottom)
S = 0.6
N_LOCAL_W = 1700
links = [(s, r, int(delay_ns[s][r])) for s in range(8) for r in range(8) if delay_ns[s][r] > 0]
fanin = {r: sorted(s for s, rr, _ in links if rr == r) for r in range(2, 8)}
COL = {2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 3}
ROW = {2: 0, 3: 2, 4: 1, 5: 1, 6: 0, 7: 2}
col_of = lambda n: COL[n]
row_of = lambda n: ROW[n]

TRK = 14
GUTTER = 270
XT = [400.0]
for _ in range(3):
    XT.append(XT[-1] + S * (N_LOCAL_W - 70) + GUTTER)      # global x of each column's input terminals
OX = [x - S * 70 for x in XT]
RIGHT = [ox + S * N_LOCAL_W for ox in OX]
GAP_TOP, GAP_H, MOD_H = [], 14 * 8, S * 550
y = 110.0
for r in range(3):
    GAP_TOP.append(y)
    y += GAP_H + MOD_H
OY = [GAP_TOP[r] + GAP_H - S * 20 for r in range(3)]
MOD_BOTTOM = [GAP_TOP[r] + GAP_H + MOD_H for r in range(3)]
LEGEND_Y = MOD_BOTTOM[2] + 50
W = RIGHT[3] + 3 * 16 + 70
H = LEGEND_Y + 300


def G(n, x, y_):
    return OX[col_of(n)] + S * x, OY[row_of(n)] + S * y_


SENS = {0: dict(cx=130.0, cy=GAP_TOP[0] + GAP_H + 120, lane=250.0, label='i1 · pulser1 (U1)',
                note='sensor click at 1 ns'),
        1: dict(cx=130.0, cy=GAP_TOP[2] + GAP_H + 120, lane=266.0, label='i2 · pulser2 (U4)',
                note='click at 1 ns + dt')}


def lane_x(s):
    if s in SENS:
        return SENS[s]['lane']
    return RIGHT[col_of(s)] + 20 + 16 * row_of(s)


def p0(s):
    if s in SENS:
        return SENS[s]['cx'], SENS[s]['cy'] - 20 - 30
    return G(s, 1640, 180)


chan_y = {}
for row in range(3):
    j = 0
    for col in range(4):
        for n in range(2, 8):
            if COL[n] != col or ROW[n] != row:
                continue
            for i in reversed(range(len(fanin[n]))):      # outermost (lowest) slot gets the highest channel
                j += 1
                chan_y[(fanin[n][i], n)] = GAP_TOP[row] + TRK * j


def build():
    o = [f'<?xml version="1.0" encoding="UTF-8"?>',
         f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}" '
         f'font-family="Inter, Helvetica, Arial, sans-serif">',
         '<defs>' + ''.join(
             f'<marker id="ah{c}" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5" markerHeight="5" orient="auto">'
             f'<path d="M0 0L10 5L0 10Z" fill="{col}"/></marker>' for c, col in LOOP.items()) + '</defs>',
         f'<rect width="{W:.0f}" height="{H:.0f}" fill="#fff"/>',
         T(40, 44, 'Six transponders · laid out like the network graph, each drawn as a copy of the single-transponder schematic', 26, BLACK, 600),
         T(40, 70, 'Every box is one copy of single_transponder_simple.pdf; the pulsers are replaced by the links. '
                   'Wire colour = sending node. Blue monospace = variable names in toy_example.py / notebook N06.',
           13, GREY)]

    for n in range(2, 8):
        ox, oy = OX[col_of(n)], OY[row_of(n)]
        srcs = [(names[s], int(delay_ns[s][n])) for s in fanin[n]]
        o.append(f'<g transform="translate({ox:.1f} {oy:.1f}) scale({S})">{module(n, srcs)}</g>')

    # sensors
    for s, d in SENS.items():
        cx, cy = d['cx'], d['cy']
        c = WIRE[names[s]]
        o.append(f'<circle cx="{cx}" cy="{cy}" r="20" fill="#fff" stroke="{INK}" stroke-width="2.6"/>')
        o.append(P(f'M{cx} {cy + 12}V{cy - 11} M{cx - 6} {cy - 3}l6 -8 6 8', w=2.4))
        o.append(P(f'M{cx} {cy + 20}V{cy + 60}'))
        o.append(ground(cx, cy + 60))
        o.append(T(cx, cy + 112, d['label'], 15, c, 700, 'middle'))
        o.append(T(cx, cy + 130, d['note'], 12, GREY, anchor='middle'))
        n_out = int((delay_ns[s] > 0).sum())
        o.append(tag(cx, cy + 148, f'n = {n_out}, same Ib2·split rule', anchor='middle', size=11))
        o.append(P(f'M{cx} {cy - 20}V{cy - 50}', w=2.6, extra=f'stroke="{c}"').replace(f'stroke="{INK}"', f'stroke="{c}"'))

    # links
    for s in range(8):
        mine = [(sr, r, d) for sr, r, d in links if sr == s]
        c = WIRE[names[s]]
        x0, y0 = p0(s)
        lx = lane_x(s)
        ys = [chan_y[(s, r)] for _, r, _ in mine] + [y0]
        lo, hi = min(ys), max(ys)
        o.append(poly([(x0, y0), (lx, y0)], 2.4, c))
        o.append(poly([(lx, lo), (lx, hi)], 2.4, c))
        if lo < y0 < hi:
            o.append(dot(lx, y0, 4.2, c))
        for _, r, d in mine:
            k = len(fanin[r])
            i = fanin[r].index(s)
            yc = chan_y[(s, r)]
            tx = XT[col_of(r)] - 22 - TRK * i
            sy = OY[row_of(r)] + S * slot_y(i, k)
            o.append(poly([(lx, yc), (tx, yc), (tx, sy), (XT[col_of(r)], sy)], 2.4, c))
            o.append(f'<path d="M{XT[col_of(r)]:.1f} {sy:.1f} l-10 -5 v10z" fill="{c}"/>')
            if lo < yc < hi:
                o.append(dot(lx, yc, 4.2, c))

    # legend
    ly = LEGEND_Y
    o.append(f'<line x1="40" y1="{ly - 20}" x2="{W - 40:.0f}" y2="{ly - 20}" stroke="#d5dbe1"/>')
    o.append(T(40, ly + 10, 'Legend', 18, BLACK, 600))
    lines = [
        ('Loops (same as the single-transponder drawing):', None),
        ('① input loop: gate current into U3, fed by the links (R_in/L_in)   ② shunt loop: catches Ib1 while U3 is resistive', None),
        ('③ storage loop: I_loop circulates through Lb1 and R2 and drives the gate of U2   ④ output loop: U2 current pushed into L2/R1 and out on the links', None),
        ('', None),
        ('Things to know:', None),
        ('· split enters at one place only: each output junction. Each link gets Ib2·split/n of nominal amplitude (peak ×0.61 after OutputPulse); the rest goes to the R1 shunt.', None),
        ('· The LTspice drawing has 2 inputs (R28/L18, R29/L19). The toy network needs 3 inputs for t1, t2, t4, t5, t6, so the input branch is repeated per link.', None),
        ('· LTspice values: Ib1 ≈ 150 µA, Ib2 ≈ 160 µA, R2 = 4 Ω, R3 = 3 Ω.  Toy values: Ib1 = 12–13 µA, Ib2 = 19 µA, RT0 = 3 Ω, Rs1 = 2 Ω (notebook defaults).', None),
        ('· Sensors are drawn as the ideal pulsers of the original drawing, but the toy gives them the same output rule as a transponder (Ib2, split).', None),
        ('· L2, R1, L_nw2 and the R_in/L_in values are not variables in the toy; the delay boxes are the delay matrix entries; all ground rails are the same 0 V node.', None),
    ]
    yy = ly + 38
    for txt, _ in lines:
        bold = txt.endswith(':')
        o.append(T(40, yy, txt, 14 if bold else 13, BLACK if bold else '#3a4a5a', 600 if bold else 400))
        yy += 22 if not bold else 24
    sx = W - 40 - 8 * 110
    for i, nm in enumerate(names):
        o.append(f'<rect x="{sx + 110 * i:.0f}" y="{ly + 2}" width="22" height="8" fill="{WIRE[nm]}"/>')
        o.append(T(sx + 110 * i + 30, ly + 11, f'from {nm}', 12, GREY))
    o.append('</svg>')
    return '\n'.join(o)


def render(svg_path):
    from playwright.sync_api import sync_playwright
    svg = svg_path.read_text()
    page_html = f'<html><body style="margin:0">{svg}</body></html>'
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={'width': int(W), 'height': int(H)}, device_scale_factor=1.5)
        pg.set_content(page_html)
        pg.screenshot(path=str(svg_path.with_suffix('.png')), full_page=False)
        pg.pdf(path=str(svg_path.with_suffix('.pdf')), width=f'{W}px', height=f'{H}px', print_background=True,
               page_ranges='1')
        b.close()


if __name__ == '__main__':
    out = HERE / 'N06_network_schematic.svg'
    out.write_text(build(), encoding='utf-8')
    print('wrote', out, f'({W:.0f} x {H:.0f})')
    if '--render' in sys.argv:
        render(out)
        print('wrote png + pdf')
