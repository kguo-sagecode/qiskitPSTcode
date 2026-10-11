"""
Bar chart of n against P(target) for peak state transfer on X_n at
r = r*(n) (report, Section 8.4): Aer (noiseless) and ibm_kingston hardware
in the cz and fractional bases. The values in DATA are the results of
xn_aer_per_vertex_optimal_r.py and the two hardware sweeps. Writes
figures/xn_pst_target_vs_n.{pdf,png}.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# (n, r*, Aer noiseless P(target), Hardware cz P(target), Hardware fractional P(target))
# r* computed for the cz survival model and reused for the fractional runs;
# the Aer column is the same for both bases.
DATA = [
    ( 2, 5, 0.899077, 0.738037, 0.766113),
    ( 3, 6, 0.836018, 0.402588, 0.490234),
    ( 4, 7, 0.780275, 0.243408, 0.321533),
    ( 5, 8, 0.708782, 0.065186, 0.173340),
    ( 6, 4, 0.435207, 0.050049, 0.062744),
    ( 7, 4, 0.698554, 0.055420, 0.072021),
    ( 8, 5, 0.710666, 0.012451, 0.022949),
    ( 9, 6, 0.655381, 0.000977, 0.009277),
    (10, 7, 0.581118, 0.000000, 0.000732),
    (11, 8, 0.504908, 0.000000, 0.000000),
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
    ax.set_title("Peak state transfer in the $X_n$ family", fontsize=13)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.95)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out_pdf = FIGURES / "xn_pst_target_vs_n.pdf"
    out_png = FIGURES / "xn_pst_target_vs_n.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
