"""
Exact noiseless distributions for the six C_4 hardware circuits
(UnitaryGate and PauliEvolutionGate, at t = pi/2, t = pi/4, and two blocks
of pi/4) after transpilation with fractional gates enabled. For the two
t = pi/2 PST circuits they agree with the cz-basis distributions of
c4_aer_device_distributions.py to six decimal places (report, Section 5.4).

Same method as c4_aer_device_distributions.py (extract the active
subcircuit of the transpiled circuit, compute the exact statevector, and
marginalize over routing qubits), but transpiled against ibm_kingston with
use_fractional_gates=True and post-processed with FoldRzzAngle.

Requires an IBM Quantum account to fetch the backend target; nothing is
submitted.
"""

from __future__ import annotations

from math import pi

import numpy as np
from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, UnitaryGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.transpiler.passes.basis import FoldRzzAngle


N = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
NOT_QUBIT = 0
TARGET_BIT = "0100"
SINGLE_EXC = {"0001", "0010", "0100", "1000"}


def hamiltonian() -> SparsePauliOp:
    terms = []
    for i, j in EDGES:
        for P in ("X", "Y"):
            lab = ["I"] * N
            lab[i] = P
            lab[j] = P
            terms.append(("".join(reversed(lab)), 0.5))
    return SparsePauliOp.from_list(terms)


def opt_a(t: float, blocks: int = 1) -> QuantumCircuit:
    U = expm(-1j * t * hamiltonian().to_matrix())
    qc = QuantumCircuit(N)
    qc.x(NOT_QUBIT)
    for _ in range(blocks):
        qc.append(UnitaryGate(U, label="U"), range(N))
    qc.measure_all()
    return qc


def opt_b(t: float, reps: int, blocks: int = 1) -> QuantumCircuit:
    evo = PauliEvolutionGate(
        hamiltonian(), time=t, synthesis=SuzukiTrotter(order=2, reps=reps)
    )
    qc = QuantumCircuit(N)
    qc.x(NOT_QUBIT)
    for _ in range(blocks):
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


def extract_active_subcircuit(isa: QuantumCircuit) -> QuantumCircuit:
    active = sorted({isa.find_bit(q).index for instr in isa.data for q in instr.qubits
                     if instr.operation.name != "barrier"})
    phys_to_new = {p: i for i, p in enumerate(active)}
    n_cl = sum(c.size for c in isa.cregs) if isa.cregs else 0
    new = QuantumCircuit(len(active), n_cl)
    for instr in isa.data:
        new_q = [phys_to_new[isa.find_bit(q).index] for q in instr.qubits]
        new_c = [isa.find_bit(c).index for c in instr.clbits]
        new.append(instr.operation, new_q, new_c)
    return new


def exact_distribution(isa: QuantumCircuit) -> tuple[dict[str, float], dict]:
    ops = dict(isa.count_ops())
    info = {"depth": isa.depth(), "ops": ops,
            "n2q": ops.get("cz", 0) + ops.get("rzz", 0)}
    small = extract_active_subcircuit(isa)
    info["n_active"] = small.num_qubits
    unitary = QuantumCircuit(small.num_qubits)
    meas_map: dict[int, int] = {}
    for instr in small.data:
        op = instr.operation
        if op.name == "measure":
            meas_map[small.find_bit(instr.clbits[0]).index] = \
                small.find_bit(instr.qubits[0]).index
        elif op.name == "barrier":
            continue
        else:
            unitary.append(op, [small.find_bit(q).index for q in instr.qubits])
    sv = Statevector.from_instruction(unitary)
    probs = np.abs(np.asarray(sv.data)) ** 2
    out = {format(i, f"0{N}b"): 0.0 for i in range(2 ** N)}
    for idx in range(2 ** small.num_qubits):
        cl_value = 0
        for cbit, qbit in meas_map.items():
            if (idx >> qbit) & 1:
                cl_value |= (1 << cbit)
        out[format(cl_value, f"0{N}b")] += float(probs[idx])
    return out, info


def main() -> None:
    backend = QiskitRuntimeService().backend("ibm_kingston", use_fractional_gates=True)
    print(f"Device transpile target: {backend.name} (fractional gates)")
    print("(transpile-only, no hardware submission)")
    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend, seed_transpiler=0
    )

    configs = [
        ("Option A, t=pi/2, fractional",          opt_a(pi / 2)),
        ("Option B (r=8), t=pi/2, fractional",    opt_b(pi / 2, 8)),
        ("Option A, t=pi/4, fractional",          opt_a(pi / 4)),
        ("Option B (r=4), t=pi/4, fractional",    opt_b(pi / 4, 4)),
        ("Option A, (pi/4)x2, fractional",        opt_a(pi / 4, 2)),
        ("Option B (r=4 each), (pi/4)x2, fract.", opt_b(pi / 4, 4, 2)),
    ]
    dists = []
    for label, qc in configs:
        isa = fold_and_clean(pm.run(qc))
        d, info = exact_distribution(isa)
        dists.append(d)
        print(f"\n=== {label} ===")
        print(f"  ISA: depth={info['depth']} 2q={info['n2q']} "
              f"n_active={info['n_active']} ops={info['ops']}")
        for bit in sorted(d.keys()):
            marker = "  single-exc" if bit in SINGLE_EXC else ""
            marker += "  *TARGET*" if bit == TARGET_BIT else ""
            print(f"  {bit}   {d[bit]:.6f}{marker}")
        print(f"  -- in-subspace sum: {sum(d[b] for b in SINGLE_EXC):.6f}")

    print("\nLaTeX rows: bit & A pi/2 & B pi/2 r8 & A pi/4 & B pi/4 r4 & A x2 & B x2 r4"
          "  (fractional)")
    for bit in sorted(dists[0].keys()):
        print(f"\\texttt{{{bit}}} & " +
              " & ".join(f"${d[bit]:.6f}$" for d in dists) + r" \\")


if __name__ == "__main__":
    main()
