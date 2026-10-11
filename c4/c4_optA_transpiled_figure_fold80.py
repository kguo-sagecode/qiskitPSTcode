"""
Drawings of the two t = pi/2 C_4 PST circuits (UnitaryGate, and
PauliEvolutionGate with r = 8) after transpilation to ibm_kingston in the
default cz basis, as in the transpiled-circuit figure of Section 5.3 of the
report. Writes

  figures/c_4_pi_on_2_transpiled_optA_fold80.{png,pdf}
  figures/c_4_pi_on_2_transpiled_optB_matched_fold80.{png,pdf}

Requires an IBM Quantum account to fetch the backend target; nothing is
submitted.
"""

from __future__ import annotations

from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, UnitaryGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService


N = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
NOT_QUBIT = 0
T = pi / 2
FOLD = 80

FIGURES = Path(__file__).parent / "figures"


def hamiltonian() -> SparsePauliOp:
    terms = []
    for i, j in EDGES:
        for P in ("X", "Y"):
            lab = ["I"] * N
            lab[i] = P
            lab[j] = P
            terms.append(("".join(reversed(lab)), 0.5))
    return SparsePauliOp.from_list(terms)


def option_a() -> QuantumCircuit:
    H = hamiltonian()
    U = expm(-1j * T * H.to_matrix())
    qc = QuantumCircuit(N)
    qc.x(NOT_QUBIT)
    qc.append(UnitaryGate(U, label="U"), range(N))
    qc.measure_all()
    return qc


def option_b(reps: int = 8) -> QuantumCircuit:
    evo = PauliEvolutionGate(
        hamiltonian(), time=T, synthesis=SuzukiTrotter(order=2, reps=reps)
    )
    qc = QuantumCircuit(N)
    qc.x(NOT_QUBIT)
    qc.append(evo, range(N))
    qc.measure_all()
    return qc


def transpile_to_device(qc: QuantumCircuit, backend) -> QuantumCircuit:
    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend, seed_transpiler=0
    )
    return pm.run(qc)


def save(isa: QuantumCircuit, name: str) -> None:
    ops = dict(isa.count_ops())
    two_q = sum(ops.get(k, 0) for k in ("ecr", "cz", "cx", "swap"))
    print(f"{name}: depth={isa.depth()}, total_ops={sum(ops.values())}, 2q={two_q}")

    fig = isa.draw(
        output="mpl",
        fold=FOLD,
        idle_wires=False,
        style="iqp-dark",
    )
    out_png = FIGURES / f"{name}.png"
    out_pdf = FIGURES / f"{name}.pdf"
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out_pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  wrote {out_png}")
    print(f"  wrote {out_pdf}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    backend = QiskitRuntimeService().backend("ibm_kingston")
    print(f"Transpile target: {backend.name}  "
          f"(transpile-only, no hardware submission)")
    save(transpile_to_device(option_a(), backend),
         "c_4_pi_on_2_transpiled_optA_fold80")
    save(transpile_to_device(option_b(reps=8), backend),
         "c_4_pi_on_2_transpiled_optB_matched_fold80")


if __name__ == "__main__":
    main()
