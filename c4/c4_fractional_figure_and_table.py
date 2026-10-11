"""
Transpiled gate counts for Section 5.3 of the report:

  - Table: depth / total / cz / rzz / 2q for the two t = pi/2 PST circuits
    (UnitaryGate, and PauliEvolutionGate with r = 8), each transpiled
    against ibm_kingston with and without fractional gates. Totals exclude
    the final measurements and the measure_all barrier.
  - Drawings (fold=80) of the fractional-basis transpiled circuits:
      figures/c_4_pi_on_2_transpiled_optA_fractional_fold80.{png,pdf}
      figures/c_4_pi_on_2_transpiled_optB_fractional_fold80.{png,pdf}
  - The current median calibration errors (cz, rzz, sx) of the device.

Requires an IBM Quantum account to fetch the backend target; nothing is
submitted.
"""

from __future__ import annotations

from math import pi
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, UnitaryGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.transpiler.passes.basis import FoldRzzAngle


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
    U = expm(-1j * T * hamiltonian().to_matrix())
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


def fold_and_clean(qct: QuantumCircuit) -> QuantumCircuit:
    folded = PassManager([FoldRzzAngle()]).run(qct)
    out = folded.copy_empty_like()
    for inst in folded.data:
        if inst.operation.name == "global_phase":
            out.global_phase += inst.operation.params[0]
        else:
            out.append(inst)
    return out


def table_row(qct: QuantumCircuit) -> dict:
    ops = dict(qct.count_ops())
    total = sum(v for k, v in ops.items() if k not in ("measure", "barrier"))
    return {
        "depth": qct.depth(lambda inst: inst.operation.name not in ("measure", "barrier")),
        "total": total,
        "cz": ops.get("cz", 0),
        "rzz": ops.get("rzz", 0),
        "two_q": ops.get("cz", 0) + ops.get("rzz", 0) + ops.get("swap", 0),
    }


def save_figure(isa: QuantumCircuit, name: str) -> None:
    fig = isa.draw(output="mpl", fold=FOLD, idle_wires=False, style="iqp-dark")
    for ext in ("png", "pdf"):
        out = FIGURES / f"{name}.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  wrote {out}")
    plt.close(fig)


def median_errors(backend) -> dict:
    out = {}
    for gate in ("cz", "rzz", "sx", "x"):
        if gate not in backend.target.operation_names:
            continue
        errs = [
            props.error
            for props in backend.target[gate].values()
            if props is not None and props.error is not None
        ]
        if errs:
            out[gate] = median(errs)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    service = QiskitRuntimeService()
    king_cz = service.backend("ibm_kingston", use_fractional_gates=False)
    king_fr = service.backend("ibm_kingston", use_fractional_gates=True)
    print("(transpile-only, no hardware submission)\n")

    print("Median calibration errors (live target):")
    for g, e in median_errors(king_fr).items():
        print(f"  {g}: {e:.2e}")

    pm_cz = generate_preset_pass_manager(
        optimization_level=3, backend=king_cz, seed_transpiler=0
    )
    pm_fr = generate_preset_pass_manager(
        optimization_level=3, backend=king_fr, seed_transpiler=0
    )

    circuits = [("UnitaryGate", option_a()), ("PauliEvolutionGate r=8", option_b(8))]

    print(f"\n{'Configuration':<26s}{'basis':<14s}"
          f"{'depth':>6s}{'total':>7s}{'cz':>5s}{'rzz':>5s}{'2q':>5s}")
    print("-" * 68)
    figs = {
        "UnitaryGate": "c_4_pi_on_2_transpiled_optA_fractional_fold80",
        "PauliEvolutionGate r=8": "c_4_pi_on_2_transpiled_optB_fractional_fold80",
    }
    for label, qc in circuits:
        for basis, pm in (("cz", pm_cz), ("fractional", pm_fr)):
            isa = pm.run(qc)
            if basis == "fractional":
                isa = fold_and_clean(isa)
                save_figure(isa, figs[label])
            r = table_row(isa)
            print(f"{label:<26s}{basis:<14s}"
                  f"{r['depth']:>6d}{r['total']:>7d}{r['cz']:>5d}{r['rzz']:>5d}{r['two_q']:>5d}")


if __name__ == "__main__":
    main()
