import numpy as np, matplotlib
import os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from three_transponder import build_snn2, simulate, ISW_G

SIGN = {"bias11->N024": -1, "pulser2->N024": -1, "U3g": -1,
        "shbias3": -1, "cpA": +1}                                          
MOD, LTC = "#2C8572", "#AF4B3C"

def main(bv=1.5, dt=6.0):
    net = build_snn2(bv, dt)
    probes = list(SIGN.keys())
    t, out = simulate(net, tstop=28e-9, h=1e-12, probe=probes)
    d = np.load("gate4.npz") if (bv == 1.5 and os.path.exists("gate4.npz")) else None
    key = f"{dt}"
    real = {}
    if d is not None and f"{key}|t" in d.files:
        real["t"] = d[f"{key}|t"]
        real["U3g"] = d[f"{key}|I(L18)"] + d[f"{key}|I(L19)"]               
        real["shbias3"] = d[f"{key}|I(L1)"]
        real["cpA"] = d[f"{key}|I(Lb1)"]

    panels = [
        ("bias11->N024", "R28 + L18 (pulser1 -> U3 gate)"),
        ("pulser2->N024", "R29 + L19 (pulser2 -> U3 gate)"),
        ("U3g", "g of U3 (combined gate current)"),
        ("shbias3", "L1 / R3 (A's shunt loop)"),
        ("cpA", "R2 + Lb1 (A's coupling, same current -- they're in series)"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 1.6*len(panels)), sharex=True)
    for ax, (key_, title) in zip(axes, panels):
        y = out[key_] * SIGN[key_]                                          
        ax.plot(t*1e9, y*1e6, color=MOD, lw=1.4, label="model (sign-corrected)")
        if key_ in real:
            ax.plot(real["t"]*1e9, real[key_]*1e6, "--", color=LTC, lw=1.2, label="LTspice (measured)")
        if key_ == "cpA":
            ax.axhline(ISW_G*1e6, color=LTC, ls=":", lw=0.9)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel("uA", fontsize=8)
        ax.legend(frameon=False, fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("time (ns)")
    fig.suptitle(f"Requested branches, sign-corrected (bv={bv} V, dt={dt:+.1f} ns)", y=1.0)
    fig.tight_layout()
    fig.savefig(f"requested_{bv}v_{dt:+.1f}ns.png", dpi=140, bbox_inches="tight")
    print(f"wrote requested_{bv}v_{dt:+.1f}ns.png")

if __name__ == "__main__":
    main()