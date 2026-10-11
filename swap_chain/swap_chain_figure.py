"""Render the n=10 SWAP-chain circuit as a figure (report, Section 9.1).

Produces figures/swap_chain_n10.pdf, the standard ten-qubit chain
X(q_0); SWAP(0,1); ...; SWAP(8,9); measure_all.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit


N = 10
FIGURES = Path(__file__).parent / "figures"


def build_chain(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"swap_chain_n{n}")
    qc.x(0)
    for k in range(n - 1):
        qc.swap(k, k + 1)
    qc.measure_all()
    return qc


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    qc = build_chain(N)
    fig = qc.draw(output="mpl", fold=-1, idle_wires=True, style="iqp")
    out_pdf = FIGURES / f"swap_chain_n{N}.pdf"
    out_png = FIGURES / f"swap_chain_n{N}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
