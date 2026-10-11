"""
Bar chart comparing measured hardware P(target) across the three
constructions (report, Section 9.4), indexed by transfer distance d (number
of qubits on the path from the source to the target, including both):

  - SWAP chain on n qubits           : d = n
  - Hamming-weighted path HP(Q_n)    : d = n + 1
  - X_n peak ST                      : d = 2n - 1   (odd d only)

The two quantum-walk constructions are shown in both the default cz
basis and the fractional basis; the SWAP chain transpiles identically
in the two bases, so it has a single bar.

All numbers are 4096-shot runs on ibm_kingston.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# (d, SWAP, HP cz, HP frac, X_n cz, X_n frac)   -- None where no data
ROWS = [
    ( 2, 0.754395, None,     None,     None,     None),
    ( 3, 0.836182, 0.915527, 0.896729, 0.738037, 0.766113),
    ( 4, 0.814941, 0.805420, 0.849365, None,     None),
    ( 5, 0.773438, 0.708496, 0.735107, 0.402588, 0.490234),
    ( 6, 0.767822, 0.495361, 0.553955, None,     None),
    ( 7, 0.760742, 0.414307, 0.502441, 0.243408, 0.321533),
    ( 8, 0.738281, 0.256836, 0.284912, None,     None),
    ( 9, 0.718506, 0.078857, 0.223389, 0.065186, 0.173340),
    (10, 0.690918, 0.101562, 0.182373, None,     None),
    (11, 0.704834, 0.022217, 0.051758, 0.050049, 0.062744),
    (12, 0.655029, 0.003418, 0.057617, None,     None),
    (13, 0.645752, 0.002197, 0.029297, 0.055420, 0.072021),
    (14, 0.603760, 0.064453, 0.010986, None,     None),
    (15, 0.600830, 0.000488, 0.000732, 0.012451, 0.022949),
    (16, 0.588623, 0.008789, 0.001953, None,     None),
    (17, 0.552002, 0.000000, 0.000488, 0.000977, 0.009277),
    (18, 0.469482, 0.002197, 0.001221, None,     None),
    (19, 0.421143, 0.000000, 0.000000, 0.000000, 0.000732),
    (20, 0.394531, 0.001953, 0.000244, None,     None),
    (21, 0.389404, 0.000488, 0.000000, 0.000000, 0.000000),
]

FIGURES = Path(__file__).parent / "figures"


def col(i):
    return np.array([(r[i] if r[i] is not None else np.nan) for r in ROWS], dtype=float)


def main() -> None:
    ds = np.array([r[0] for r in ROWS])
    swap = col(1)
    hp_cz, hp_fr = col(2), col(3)
    xn_cz, xn_fr = col(4), col(5)

    x = np.arange(len(ds))
    width = 0.16

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(x - 2 * width, swap,  width, label="SWAP chain",
           color="#0072B2", edgecolor="black", linewidth=0.3)
    ax.bar(x - width,     hp_cz, width, label="Hamming path (cz)",
           color="#D55E00", edgecolor="black", linewidth=0.3)
    ax.bar(x,             hp_fr, width, label="Hamming path (frac.)",
           color="#F0A868", edgecolor="black", linewidth=0.3)
    ax.bar(x + width,     xn_cz, width, label=r"$X_n$ peak ST (cz)",
           color="#009E73", edgecolor="black", linewidth=0.3)
    ax.bar(x + 2 * width, xn_fr, width, label=r"$X_n$ peak ST (frac.)",
           color="#7FD4B8", edgecolor="black", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(ds)
    ax.set_xlabel(r"Transfer distance $d$ (qubits between source and target, inclusive)",
                  fontsize=12)
    ax.set_ylabel(r"$P(\mathrm{target})$ on ibm_kingston, 4096 shots",
                  fontsize=12)
    ax.set_title("Hardware target probability vs. transfer distance, three constructions",
                 fontsize=13)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95, ncol=2)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.rcParams["text.usetex"] = False
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out_pdf = FIGURES / "transfer_distance_comparison.pdf"
    out_png = FIGURES / "transfer_distance_comparison.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
