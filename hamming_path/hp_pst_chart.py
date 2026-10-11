"""
Bar chart of n against P(target) for PST on the Hamming-weighted paths
HP_n at r = r*(n) (report, Section 7.3): Aer (noiseless) and ibm_kingston
hardware in the cz and fractional bases. The values in DATA are the
results of hamming_path_aer_per_vertex_optimal_r.py and the two hardware
sweeps. Writes figures/hp_pst_target_vs_n.{pdf,png}.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# (n, r*, Aer noiseless P(target), Hardware cz P(target), Hardware fractional P(target))
# r* computed for the cz survival model at p_2q = 2.0e-3 and reused for the
# fractional runs; the Aer column is the same for both bases.
DATA = [
    ( 2,  3, 0.994421, 0.915527, 0.896729),
    ( 3,  4, 0.988575, 0.805420, 0.849365),
    ( 4,  4, 0.967533, 0.708496, 0.735107),
    ( 5,  5, 0.967695, 0.495361, 0.553955),
    ( 6,  5, 0.933560, 0.414307, 0.502441),
    ( 7,  6, 0.933957, 0.256836, 0.284912),
    ( 8,  7, 0.931470, 0.078857, 0.223389),
    ( 9,  7, 0.881124, 0.101562, 0.182373),
    (10,  8, 0.879425, 0.022217, 0.051758),
    (11,  8, 0.810534, 0.003418, 0.057617),
    (12, 10, 0.871119, 0.002197, 0.029297),
    (13,  3, 0.535430, 0.064453, 0.010986),
    (14, 10, 0.732475, 0.000488, 0.000732),
    (15,  4, 0.362132, 0.008789, 0.001953),
    (16, 12, 0.735073, 0.000000, 0.000488),
    (17,  4, 0.424774, 0.002197, 0.001221),
    (18, 12, 0.559489, 0.000000, 0.000000),
    (19,  4, 0.402472, 0.001953, 0.000244),
    (20,  5, 0.162311, 0.000488, 0.000000),
]

FIGURES = Path(__file__).parent / "figures"


def main() -> None:
    ns = [d[0] for d in DATA]
    aer = [d[2] for d in DATA]
    hw_cz = [d[3] for d in DATA]
    hw_fr = [d[4] for d in DATA]
    x = np.arange(len(ns))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(x - width, aer, width, label="Aer simulator",
           color="#5d3a9b", edgecolor="black", linewidth=0.4)
    ax.bar(x, hw_cz, width, label="ibm_kingston (cz)",
           color="#e66100", edgecolor="black", linewidth=0.4)
    ax.bar(x + width, hw_fr, width, label="ibm_kingston (fractional)",
           color="#289b32", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(ns)
    ax.set_xlabel(r"$n$", fontsize=12)
    ax.set_ylabel(r"$P(\mathrm{target})$", fontsize=12)
    ax.set_title("PST in Hamming-weighted paths", fontsize=13)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.95)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.rcParams["text.usetex"] = False  # avoid TeX dependency
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out_pdf = FIGURES / "hp_pst_target_vs_n.pdf"
    out_png = FIGURES / "hp_pst_target_vs_n.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
