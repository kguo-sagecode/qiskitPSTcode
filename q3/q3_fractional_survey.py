"""
Transpilation and noiseless simulation of the two Q_3 (3-cube) hardware
circuits, in the default cz basis and with fractional gates enabled,
against the ibm_kingston targets.

Configurations (the two run on hardware; report, Section 6):
  - PauliEvolutionGate r=8, t = pi/2  (PST, target |e_7>)
  - PauliEvolutionGate r=4, t = pi/4  (uniform mixing)

For each: device transpile (opt 3, seed 0) to cz and fractional
(+FoldRzzAngle) targets; depth / total / cz / rzz / 2q (totals exclude
measure + barrier, as in the report's Q_3 gate-count table); exact
noiseless per-vertex probabilities for both bases, checked against each
other.

Requires an IBM Quantum account to fetch the backend target; nothing is
submitted. Results: printed table + q3_fractional_survey.json.
"""

from __future__ import annotations

import json
from math import pi

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.transpiler.passes.basis import FoldRzzAngle

N = 8  # qubits = vertices of Q_3
EDGES = [(k, k ^ (1 << b)) for k in range(8) for b in range(3) if k < (k ^ (1 << b))]
RESULTS_FILE = "q3_fractional_survey.json"


def hamiltonian() -> SparsePauliOp:
    terms = []
    for i, j in EDGES:
        for P in ("X", "Y"):
            lab = ["I"] * N
            lab[i] = P
            lab[j] = P
            terms.append(("".join(reversed(lab)), 0.5))
    return SparsePauliOp.from_list(terms)


def opt_b(t: float, reps: int) -> QuantumCircuit:
    evo = PauliEvolutionGate(
        hamiltonian(), time=t, synthesis=SuzukiTrotter(order=2, reps=reps)
    )
    qc = QuantumCircuit(N)
    qc.x(0)
    qc.append(evo, range(N))
    qc.measure_all()
    return qc


def fold_and_clean(qct: QuantumCircuit) -> QuantumCircuit:
    folded = PassManager([FoldRzzAngle()]).run(qct)
    out = qct.copy_empty_like()
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
        "depth": qct.depth(lambda i: i.operation.name not in ("measure", "barrier")),
        "total": total,
        "cz": ops.get("cz", 0),
        "rzz": ops.get("rzz", 0),
        "two_q": ops.get("cz", 0) + ops.get("rzz", 0),
    }


def per_vertex(isa: QuantumCircuit) -> dict[int, float]:
    active = sorted({isa.find_bit(q).index for ins in isa.data for q in ins.qubits
                     if ins.operation.name != "barrier"})
    ph2new = {p: i for i, p in enumerate(active)}
    uni = QuantumCircuit(len(active))
    meas: dict[int, int] = {}
    for ins in isa.data:
        if ins.operation.name == "measure":
            meas[isa.find_bit(ins.clbits[0]).index] = \
                ph2new[isa.find_bit(ins.qubits[0]).index]
        elif ins.operation.name != "barrier":
            uni.append(ins.operation, [ph2new[isa.find_bit(q).index] for q in ins.qubits])
    probs = np.abs(Statevector.from_instruction(uni).data) ** 2
    out = {k: 0.0 for k in range(N)}
    for idx, pr in enumerate(probs):
        cl = 0
        for cbit in range(N):
            if (idx >> meas[cbit]) & 1:
                cl |= 1 << cbit
        if cl and (cl & (cl - 1)) == 0:
            out[cl.bit_length() - 1] += pr
    return out


def main() -> None:
    service = QiskitRuntimeService()
    king_cz = service.backend("ibm_kingston", use_fractional_gates=False)
    king_fr = service.backend("ibm_kingston", use_fractional_gates=True)
    pm_cz = generate_preset_pass_manager(
        optimization_level=3, backend=king_cz, seed_transpiler=0
    )
    pm_fr = generate_preset_pass_manager(
        optimization_level=3, backend=king_fr, seed_transpiler=0
    )
    print("(transpile + Aer only, no hardware submission)\n")

    cases = [
        ("B8, t=pi/2 (PST)",    opt_b(pi / 2, 8)),
        ("B4, t=pi/4 (mixing)", opt_b(pi / 4, 4)),
    ]
    results = {}
    print(f"{'Configuration':<22s} {'basis':<12s} {'depth':>6s} {'total':>6s} "
          f"{'cz':>5s} {'rzz':>5s} {'2q':>5s}")
    for label, qc in cases:
        pv = {}
        for basis, pm in (("cz", pm_cz), ("fractional", pm_fr)):
            isa = pm.run(qc)
            if basis == "fractional":
                isa = fold_and_clean(isa)
                angles = [float(i.operation.params[0]) for i in isa.data
                          if i.operation.name == "rzz"]
                assert all(0 < a <= pi / 2 + 1e-12 for a in angles), "rzz out of range"
            row = table_row(isa)
            pv[basis] = per_vertex(isa)
            results.setdefault(label, {})[basis] = {**row, "per_vertex": pv[basis]}
            print(f"{label:<22s} {basis:<12s} {row['depth']:>6d} {row['total']:>6d} "
                  f"{row['cz']:>5d} {row['rzz']:>5d} {row['two_q']:>5d}")
        max_dev = max(abs(pv['cz'][k] - pv['fractional'][k]) for k in pv['cz'])
        print(f"{'':22s} per-vertex |cz - fractional| max = {max_dev:.2e}")
        print(f"{'':22s} per-vertex (fractional): " +
              " ".join(f"{pv['fractional'][k]:.6f}" for k in range(N)))

    with open(RESULTS_FILE, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
