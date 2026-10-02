import numpy as np                                                      

S = 1.0                                                                      # .param S1=1 -> S=1**2/(0.6+0.4*1)=1
Jc = 40e9                                                                    # critical current density, A/m^2
TH = 19e-9                                                                   # film thickness, m
SHEET = 130.0                                                                # sheet resistance, ohm/square
TC, TSUB = 8.5, 4.3                                                          # critical and substrate temperature, K
W_G, W_S, W_D, W_C = 20e-9/S, 200e-9/S, 200e-9/S, 250e-9/S                   # segment widths, m
SQ_D, SQ_S, SQ_C = 400/S, 80/S, 10/S                                         # squares: drain, source, choke
SQ_G_IN = 10.0                                                               # netlist override: input nTrons (u1,u3,u4,u6)
SQ_G_OUT = 5.0/S                                                             # default sq_g: output nTrons (u2,u5,u7)
HC = 50e3                                                                    # surface heat transfer coeff, W/m^2K
HEATCAP = 4400.0                                                             # volumetric heat capacity, J/m^3K
A1, BETA = 0.4, 12.82e-6                                                     # gate-suppression gain and scale
DELTA = 0.01                                                                 # lib's anti-singularity offset
R_SH = 3.0/S                                                                 # netlist .param R_sh
R_LOOP = 2.0/S                                                               # netlist .param R_loop
L1P = 150e-9                                                                 # netlist .param L1
SQ_S2 = 80.0                                                                 # netlist .param sq_s2
IND = 1.38e-12 * SHEET / TC                                                  # kinetic inductance, H per square
PSI = SHEET * (Jc * TH) ** 2 / (HC * (TC - TSUB))                            # Stekly parameter
KAPPA = 2.44e-8 * TC / (SHEET * TH)                                          # thermal conductivity
VO = np.sqrt(HC * KAPPA / TH) / HEATCAP                                      # hotspot boundary velocity
ISW_G = Jc * W_G * TH                                                        # gate critical current (15.200 uA)
ISW_C = Jc * W_C * TH                                                        # choke critical current (190 uA)
ISW_S = Jc * W_S * TH                                                        # source segment critical current
ISW_D = Jc * W_D * TH                                                        # drain segment critical current
VTH_G = np.sqrt(2 / PSI) * ISW_G                                             # Vthresh_g = minSquares*sheetRes*Ihs_g
VTH_C = np.sqrt(2 / PSI) * ISW_C                                             # same identity for the channel
RN_C2 = SHEET * SQ_C / 2                                                     # Rnorm_c/2, each choke-half's ceiling
RN_S = SHEET * SQ_S                                                          # Rnorm_s, source segment's ceiling
RN_D = SHEET * SQ_D                                                          # Rnorm_d, drain segment's ceiling
C_G = W_G / (SHEET * VO)                                                     # C1, gate growth capacitor
C_CHOKE = W_C / (SHEET * VO)                                                 # C01/C001, choke-half growth capacitors
C_SRC = W_S / (SHEET * VO)                                                   # C02, source-segment growth capacitor
C_DRN = W_D / (SHEET * VO)                                                   # C002, drain-segment growth capacitor
LB = L1P - (SQ_G_OUT + SQ_S2) * IND                                          # netlist expr for Lb1/Lb2/Lb3 (148.206 nH)


def _g(i, Isw):
    """The lib's g(i) = 1/(2cos((2/3)asin(0.6|i|/Isw)) - 1), from the
    nonlinear Flux= expression on every kinetic inductor."""
    u = min(0.6 * abs(i) / Isw, 0.999999)                                   # clamp inside asin's domain
    return 1.0 / (2.0 * np.cos((2.0 / 3.0) * np.arcsin(u)) - 1.0)           # the lib's own formula, verbatim


def dphi_di(i, Lind, Isw, r, Rn):
    """Effective inductance dPhi/di of one Flux= inductor at fixed hotspot r.
    Phi = A(r)*Lind*g(i)*i ; A(r) = 1 - r/Rnorm + 0.001 (the lib's hotspot
    collapse factor). Verified against a central finite difference to 1e-10
    relative error before use."""
    A = 1.0 - r / Rn + 0.001                                                # hotspot growth collapses L toward 0
    ai = abs(i)                                                             # magnitude, the lib uses abs(x)
    u = min(0.6 * ai / Isw, 0.999999)                                       # asin domain guard
    th = (2.0 / 3.0) * np.arcsin(u)                                         # the lib's (2/3)asin(...) angle
    D = 2.0 * np.cos(th) - 1.0                                              # denominator of g(i)
    dth = (2.0 / 3.0) * (0.6 / Isw) / np.sqrt(max(1.0 - u * u, 1e-12))      # d(theta)/d|i|
    dg = 2.0 * np.sin(th) * dth / (D * D)                                   # d(g)/d|i|, quotient rule
    return A * Lind * (1.0 / D + ai * dg)                                   # product rule on Phi = A*Lind*g(i)*i


def dphi_dr(i, Lind, Isw, Rn):
    """dPhi/dr = -Lind*g(i)*i/Rnorm -- exact, since only A(r) depends on r
    and dA/dr = -1/Rnorm. This is the back-EMF term while a hotspot grows;
    omitting it was worth 3x the timing error in earlier versions."""
    return -Lind * _g(i, Isw) * i / Rn                                     # linear in i, exact derivative


def vel(i, ic):
    """Hotspot boundary velocity: the lib's B3/B03/B04/B003/B004 sources,
    (psi*(i/ic)^2 - 2) / (sqrt(max(psi*(i/ic)^2 - 1, 0)) + delta)."""
    x = PSI * (i / ic) ** 2                                                 # the lib's psi*(I/Isw)^2 term
    return (x - 2.0) / (np.sqrt(max(x - 1.0, 0.0)) + DELTA)                 # verbatim from the lib


class NTron:
    """One nTron: three branches (gate, drain, source) meeting at an
    internal `center` node, per ntron_2.lib lines 74-216. Five hotspot
    states: gate (N3), drain choke-half + segment (N003, N004), source
    choke-half + segment (N03, N04)."""

    def __init__(self, name, sq_g):
        self.name, self.sq_g = name, sq_g                                   # identity and this instance's gate width
        self.Lg = IND * sq_g                                                # Lind_g, this nTron's gate inductance
        self.Rn_g = SHEET * sq_g                                            # Rnorm_g, gate hotspot ceiling
        self.Ld, self.Ls = IND * SQ_D, IND * SQ_S                           # Lind_d, Lind_s (fixed geometry)
        self.Lc2 = IND * SQ_C / 2                                           # Lind_c/2, each choke half
        self.hs = np.zeros(5)                                               # [N3, N003, N004, N03, N04]

    def branchLR(self, ig, idr, isr):
        """Effective (inductance) for the gate, drain and source branches
        at the current hotspot state. Resistance is read directly from the
        hotspot states themselves (v(N3) etc IS the resistance, per the
        lib's B1/B01/B001 comments)."""
        N3, N003, N004, N03, N04 = self.hs                                  # unpack current hotspot resistances
        Lg = dphi_di(ig, self.Lg, ISW_G, N3, self.Rn_g)                     # gate branch inductance
        Ld = (dphi_di(idr, self.Ld, ISW_D, N004, RN_D)                      # Ld: drain segment
              + dphi_di(idr, self.Lc2, ISW_C, N003, RN_C2))                 # + Lc2: drain-side choke half
        Ls = (dphi_di(isr, self.Lc2, ISW_C, N03, RN_C2)                     # Lc1: source-side choke half
              + dphi_di(isr, self.Ls, ISW_S, N04, RN_S))                    # + Ls: source segment
        return (Lg, N3), (Ld, N003 + N004), (Ls, N03 + N04)                 # (L, R) per branch

    def crossEMF(self, ig, idr, isr, d):
        """-(dPhi/dr)(dr/dt) per branch: the back-EMF induced while each
        segment's hotspot is actively growing or shrinking."""
        eg = -dphi_dr(ig, self.Lg, ISW_G, self.Rn_g) * d[0]                 # gate segment's contribution
        ed = -(dphi_dr(idr, self.Ld, ISW_D, RN_D) * d[2]                    # drain segment's contribution
               + dphi_dr(idr, self.Lc2, ISW_C, RN_C2) * d[1])               # + drain-side choke half
        es = -(dphi_dr(isr, self.Lc2, ISW_C, RN_C2) * d[3]                  # source-side choke half
               + dphi_dr(isr, self.Ls, ISW_S, RN_S) * d[4])                 # + source segment
        return eg, ed, es                                                   # one EMF per branch

    def rates(self, ig, idr, isr):
        """Hotspot growth rates and the lib's three B2-style normal/
        superconducting flags (gate, drain side, source side)."""
        N3, N003, N004, N03, N04 = self.hs                                  # current hotspot state
        gn = (abs(ig) > ISW_G) or (abs(ig) * N3 > VTH_G)                    # B2: gate normal-state flag
        dN3 = vel(ig, ISW_G) / C_G if (gn and N3 < self.Rn_g) else 0.0      # B3: gate hotspot growth
        sup = A1 * np.exp(-(abs(ig) - ISW_G) / BETA) if abs(ig) > ISW_G else 1.0  # Bc_s: gate suppresses channel
        ic = ISW_C * sup                                                    # effective choke critical current
        dn = (abs(idr) > ic) or ((N003 + N004) * abs(idr) > VTH_C)          # B002: drain side normal-state flag
        dN003 = vel(idr, ISW_C) / C_CHOKE if (dn and N003 <= RN_C2) else 0.0  # B003: drain choke-half growth
        dN004 = vel(idr, ISW_D) / C_DRN if (dn and N003 > RN_C2 and N004 < RN_D) else 0.0  # B004: drain segment
        sn = (abs(isr) > ic) or ((N03 + N04) * abs(isr) > VTH_C)            # B02: source side normal-state flag
        dN03 = vel(isr, ISW_C) / C_CHOKE if (sn and N03 <= RN_C2) else 0.0  # B03: source choke-half growth
        dN04 = vel(isr, ISW_S) / C_SRC if (sn and N03 > RN_C2 and N04 < RN_S) else 0.0  # B04: source segment
        return np.array([dN3, dN003, dN004, dN03, dN04]), (gn, dn, sn)      # rates and the three flags

    def step(self, d, flags, h):
        """Advance hotspots by explicit Euler and apply the lib's reset switches. Done as Euler (not folded into the implicit branch solve)
        because the S1/S01/S001/S002 resets are hard discontinuities; letting an implicit or RK solver average across them was found to
        produce a spurious non-zero hotspot that never clears (a fixed latch, converged in step size, so not fixable by refining the step."""
        gn, dn, sn = flags                                                  # this step's three normal-state flags
        self.hs = self.hs + h * d                                           # explicit Euler update
        for j, cap in enumerate((self.Rn_g, RN_C2, RN_D, RN_C2, RN_S)):     # clamp each state to [0, its ceiling]
            self.hs[j] = min(max(self.hs[j], 0.0), cap)
        if not gn: self.hs[0] = 0.0                                         # S1: shorts C1 (gate resets)
        if not dn: self.hs[1] = self.hs[2] = 0.0                            # S001, S002: drain side resets
        if not sn: self.hs[3] = 0.0                                         # S01 only -- lib has no S02 (see note below)


def pulse(t, td, amp=0.25 / S, tr=2.9e-9, hold=0.1e-9, tf=2.9e-9):
    """LTspice PULSE(0 amp td tr hold tf ...) with Ncycles=1, exactly as
    V1/V4 are written in SNN_2input.asc."""
    p = t - td                                                              # time since this pulse's delay
    if p < 0: return 0.0                                                    # not started yet
    if p < tr: return amp * p / tr                                          # linear rise
    if p < tr + hold: return amp                                            # held at peak
    if p < tr + hold + tf: return amp * (1.0 - (p - tr - hold) / tf)        # linear fall
    return 0.0                                                              # finished, back to baseline

# mesh analysis of the outer circuit

class Net:
    """A network of (L, R, EMF) branches; builds its own fundamental loop
    (mesh) matrix from a spanning tree rooted at ground."""

    def __init__(self):
        self.br = []                                                        # list of (node_a, node_b, tag)
        self.nodes = {"0": 0}                                               # node name -> index, "0" is ground

    def n(self, name):
        """Look up a node's index, creating it on first use."""
        if name not in self.nodes:
            self.nodes[name] = len(self.nodes)                              # assign the next free index
        return self.nodes[name]

    def add(self, a, b, tag):
        """Add one branch between named nodes a and b; tag describes its
        electrical type (see build_snn2 for the tag formats)."""
        self.br.append((self.n(a), self.n(b), tag))                        # store by node index
        return len(self.br) - 1                                            # this branch's index

    def build_loops(self):
        """Fundamental loop matrix M: each row is one independent mesh
        current's path, as +-1 coefficients on every branch it crosses."""
        N, B = len(self.nodes), len(self.br)                               # node and branch counts
        adj = [[] for _ in range(N)]                                        # adjacency list: node -> [(nbr, branch, sign)]
        for k, (a, b, _) in enumerate(self.br):
            adj[a].append((b, k, +1))                                       # a -> b along branch k, forward
            adj[b].append((a, k, -1))                                       # b -> a along branch k, reverse
        par, pbr, seen, queue = [-1] * N, [0] * N, [False] * N, [0]        # BFS spanning tree from ground
        seen[0] = True
        tree = set()                                                        # branches used by the spanning tree
        while queue:
            u = queue.pop(0)
            for v, k, s in adj[u]:
                if not seen[v]:
                    seen[v] = True; par[v] = u; pbr[v] = (k, s); tree.add(k)
                    queue.append(v)

        def path_to_ground(x):
            """Branches (with sign) from node x back up to ground."""
            out = []
            while x != 0:
                k, s = pbr[x]; out.append((k, s)); x = par[x]
            return out

        links = [k for k in range(B) if k not in tree]                     # the co-tree branches, one per loop
        M = np.zeros((len(links), B))                                       # fundamental loop matrix
        for r, k in enumerate(links):
            a, b, _ = self.br[k]
            M[r, k] = 1.0                                                   # the link branch itself, forward
            for kk, s in path_to_ground(b): M[r, kk] += s                   # close the loop: b back to ground
            for kk, s in path_to_ground(a): M[r, kk] -= s                   # ...and ground back to a
        self.M, self.links = M, links
        return M


def build_snn2(bv=1.5, dt_ns=0.0):
    net = Net()
    tr = {}                                                                 # NTron
    for nm, sq in [("U1", SQ_G_IN), ("U4", SQ_G_IN), ("U3", SQ_G_IN),        # input-type nTrons, sq_g=10
                  ("U6", SQ_G_IN), ("U2", SQ_G_OUT), ("U5", SQ_G_OUT)]:      # output-type nTrons, sq_g=5/S
        tr[nm] = NTron(nm, sq)
    b = {}                                                                   # branch index, for lookups

    # pulse sources
    b["p1"] = net.add("0", "N048", ("src", "V1", 10e3, 5e-9))               # R5+L3 -> U1 gate
    b["p2"] = net.add("0", "N037", ("src", "V4", 10e3, 5e-9))               # R12+L9 -> U4 gate

    # every nTron's three branches, meeting at its own `center` node
    for nm, gnode, dnode in [("U1", "N048", "bias11"), ("U4", "N037", "pulser2"),
                             ("U3", "N024", "bias3"), ("U6", "N044", "bias1"),
                             ("U2", "N021", "bias4"), ("U5", "N039", "bias2")]:
        c = "c" + nm                                                        # this nTron's own center node
        b[nm + "g"] = net.add(gnode, c, ("nt", nm, "g"))                    # gate branch
        b[nm + "d"] = net.add(dnode, c, ("nt", nm, "d"))                    # drain branch
        b[nm + "s"] = net.add("0", c, ("nt", nm, "s"))                      # source branch, to ground

    # shunt loops: L + R_sh from each drain node to ground 
    for nd in ["bias11", "pulser2", "bias3", "bias1", "bias4", "bias2"]:
        b["sh" + nd] = net.add(nd, "0", ("lr", 5e-9, R_SH))                 # netlist: always 5n and R_sh

    # bias tees: V through (R27 + R26||Z0) and L16, an L/R low-pass
    for nd, V in [("bias11", 1.3), ("pulser2", 1.3), ("bias3", bv),         # bv only touches bias3
                  ("bias1", 1.3), ("bias4", 1.6), ("bias2", 1.7)]:
        b["bt" + nd] = net.add("0", nd, ("dc", V, 10e3 + 25.0, 5e-9))       # 25 = R26(50)||Z0(50)

    # pulser fan-in: both relayed pulses reach both cells' gates
    for src, gate in [("bias11", "N024"), ("pulser2", "N024"),              # U3's gate (cell A)
                      ("bias11", "N044"), ("pulser2", "N044")]:             # U6's gate (cell B)
        b[f"{src}->{gate}"] = net.add(src, gate, ("lr", 5e-9, 20.0))        # netlist: always 5n and 20 ohm

    # --- coupling branches: R2/R10 then Lb into the output nTron's gate --
    b["cpA"] = net.add("bias3", "N021", ("lr", LB, 2.0 * R_LOOP))           # A's coupling into U2
    b["cpB"] = net.add("bias1", "N039", ("lr", LB, 2.0 * R_LOOP))           # B's coupling into U5

    net.build_loops()                                                       # compute the fundamental loop matrix
    net.tr, net.bmap, net.dt_ns = tr, b, dt_ns                              # attach for simulate() to use
    return net


def dc_operating_point(net):
    """ Solve the steady state before t=0: inductors are shorts, hotspots are zero (fully superconducting), pulses are off. 
    Matches what LTspice itself does before running .tran. Starting a transient from zero current instead injects a spurious 
    turn-on surge that fires every nTron regardless of bias -- this was the actual cause of the model failing to generalise 
    across bias in earlier versions."""
    M = net.M; nB = len(net.br)
    Rv, Ev = np.zeros(nB), np.zeros(nB)                                    # per-branch resistance and source
    for k, (a, b, tag) in enumerate(net.br):
        if tag[0] == "src":
            Rv[k], Ev[k] = tag[2], 0.0                                     # pulses are off at t=0
        elif tag[0] == "dc":
            Rv[k], Ev[k] = tag[2], tag[1]                                  # bias tees are already on
        elif tag[0] == "lr":
            Rv[k], Ev[k] = tag[2], 0.0                                     # passive branch, no source
        else:
            Rv[k], Ev[k] = 1e-6, 0.0                                       # nTron: fully superconducting
    Rv = Rv + 1e-9                                                         # tiny Rser, keeps the solve non-singular
    return np.linalg.solve(M @ np.diag(Rv) @ M.T, M @ Ev)                  # DC mesh solve: R_mesh * i = E_mesh


def simulate(net, tstop=30e-9, h=1e-12, probe=("cpA",)):
    """Integrate the network from its DC operating point using trapezoidal (implicit, A-stable) steps -- required because several 
    branches have L/R time constants faster than the 1 ps step (the 10k bias branches: 0.5 ps; the nTron gate branch: 0.16 ps).
    Returns (t, out) where out[name] is that branch's current (A) over time, for every name listed in probe (see build_snn2 for 
    branch names, e.g. "cpA" is transponder A's coupling current -- compare its peak against Isw_g = 15.2e-6 to determine firing)."""
    M = net.M; nB = len(net.br); nL = M.shape[0]                           # mesh sizes
    il = dc_operating_point(net)                                           # start at the operating point, not zero
    td_ns = net.dt_ns                                                      # this run's pulse separation
    V1 = lambda t: pulse(t - 5e-9, 10e-9)                                  # T3's 5ns delay folded into V1's Td
    V4 = lambda t: pulse(t - 5e-9, 10e-9 - td_ns * 1e-9)                   # T4's delay; V4's own Td = {10n-dt}
    nstep = int(tstop / h) + 1                                             # number of fixed 1 ps steps
    out = {k: np.zeros(nstep) for k in probe}                              # requested output traces
    tt = np.zeros(nstep)                                                   # time axis
    Lv, Rv, Ev = np.zeros(nB), np.zeros(nB), np.zeros(nB)                  # per-branch values, reused each step

    for s in range(nstep):
        t = s * h                                                          # current simulation time
        ib = M.T @ il                                                      # branch currents from mesh currents
        rates, flags = {}, {}                                              # this step's hotspot rates and flags
        for nm, dev in net.tr.items():
            ig = ib[net.bmap[nm + "g"]]; idr = ib[net.bmap[nm + "d"]]; isr = ib[net.bmap[nm + "s"]]
            rates[nm], flags[nm] = dev.rates(ig, idr, isr)                 # evaluate B2/B02/B002 and B3-family

        for k, (a, bb, tag) in enumerate(net.br):                          # fill in this step's L, R, EMF
            if tag[0] == "src":
                Lv[k], Rv[k] = tag[3], tag[2]                              # pulse launcher's L and R
                Ev[k] = (V1 if tag[1] == "V1" else V4)(t)                  # this branch's instantaneous drive
            elif tag[0] == "dc":
                Lv[k], Rv[k], Ev[k] = tag[3], tag[2], tag[1]               # bias tee: fixed L, R, constant V
            elif tag[0] == "lr":
                Lv[k], Rv[k], Ev[k] = tag[1], tag[2], 0.0                  # plain passive L/R branch
            else:                                                          # an nTron branch (gate/drain/source)
                nm, term = tag[1], tag[2]
                dev = net.tr[nm]
                ig = ib[net.bmap[nm + "g"]]; idr = ib[net.bmap[nm + "d"]]; isr = ib[net.bmap[nm + "s"]]
                (Lg, Rg), (Ld, Rd), (Ls, Rs) = dev.branchLR(ig, idr, isr)  # current-dependent L, R this branch
                eg, ed, es = dev.crossEMF(ig, idr, isr, rates[nm])         # this branch's back-EMF
                Lv[k], Rv[k], Ev[k] = ((Lg, Rg, eg) if term == "g" else
                                       (Ld, Rd, ed) if term == "d" else (Ls, Rs, es))

        Lm = M @ np.diag(Lv) @ M.T                                         # mesh inductance matrix
        Rm = M @ np.diag(Rv) @ M.T                                         # mesh resistance matrix
        Em = M @ Ev                                                        # mesh EMF vector
        A = Lm / h + Rm / 2.0                                              # trapezoidal system matrix
        rhs = (Lm / h - Rm / 2.0) @ il + Em                                # trapezoidal right-hand side
        il = np.linalg.solve(A, rhs)                                       # implicit solve for the new mesh currents

        for nm, dev in net.tr.items():
            dev.step(rates[nm], flags[nm], h)                              # advance and reset each nTron's hotspots

        ibn = M.T @ il                                                     # branch currents at the new time
        tt[s] = t
        for k in probe:
            out[k][s] = ibn[net.bmap[k]]                                   # record the requested traces
    return tt, out