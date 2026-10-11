"""
Noiseless 16-state distributions for the C_4 circuits of Section 5 of the
report, computed from the same transpiled circuits that run on hardware:
each circuit is transpiled to the ibm_kingston backend (Heron r2) and then
simulated exactly.

Requires an IBM Quantum account to fetch the backend target; nothing is
submitted.

Note on the Trotter residual: Statevector.from_instruction on the
untranspiled circuit evaluates PauliEvolutionGate exactly (through
scipy.linalg.expm). After transpilation the Trotter formula is written out
as explicit gates, and for Option B at r = 8 a residual of about 5e-5
appears in the target probability, visible at six decimal places.
"""

from __future__ import annotations

from math import pi

import numpy as np
from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, UnitaryGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService


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


def opt_a(t: float, twice: bool = False) -> QuantumCircuit:
    H = hamiltonian()
    U = expm(-1j * t * H.to_matrix())
    qc = QuantumCircuit(N)
    qc.x(NOT_QUBIT)
    qc.append(UnitaryGate(U, label="U"), range(N))
    if twice:
        qc.append(UnitaryGate(U, label="U"), range(N))
    qc.measure_all()
    return qc


def opt_b(t: float, reps: int, twice: bool = False) -> QuantumCircuit:
    evo = PauliEvolutionGate(
        hamiltonian(), time=t, synthesis=SuzukiTrotter(order=2, reps=reps)
    )
    qc = QuantumCircuit(N)
    qc.x(NOT_QUBIT)
    qc.append(evo, range(N))
    if twice:
        qc.append(evo, range(N))
    qc.measure_all()
    return qc


def extract_active_subcircuit(isa: QuantumCircuit) -> QuantumCircuit:
    """Extract the subcircuit on just the active physical qubits, preserving
    the measure_all classical-bit assignment. Returns a circuit small enough
    to feed to Statevector exactly."""
    active = sorted({isa.find_bit(q).index for instr in isa.data for q in instr.qubits
                     if instr.operation.name != "barrier"})
    phys_to_new = {p: i for i, p in enumerate(active)}
    n_active = len(active)
    n_cl = sum(c.size for c in isa.cregs) if isa.cregs else 0
    new = QuantumCircuit(n_active, n_cl)
    for instr in isa.data:
        new_q = [phys_to_new[isa.find_bit(q).index] for q in instr.qubits]
        new_c = [isa.find_bit(c).index for c in instr.clbits]
        new.append(instr.operation, new_q, new_c)
    return new


def device_distribution(qc: QuantumCircuit, backend) -> tuple[dict[str, float], dict]:
    """Transpile qc to `backend` (Heron heavy-hex) at opt_level=3, then compute
    the EXACT noiseless distribution over the 4-bit classical register from
    the device-transpiled circuit via Statevector. Returns (prob dict, info)."""
    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend, seed_transpiler=0
    )
    isa = pm.run(qc)
    ops = dict(isa.count_ops())
    n2q = sum(ops.get(g, 0) for g in ("cz", "ecr", "cx", "swap"))
    info = {"depth": isa.depth(), "total_ops": sum(ops.values()),
            "n2q": n2q, "ops": ops}
    # Extract the active-qubits subcircuit, then build a measurement-free
    # version on a fresh quantum-only register so Statevector accepts it.
    small = extract_active_subcircuit(isa)
    info["n_active"] = small.num_qubits
    # Walk the small circuit's gates and record (a) the unitary part on a
    # measurement-free register, (b) which active-qubit-index is measured by
    # each classical bit.
    unitary = QuantumCircuit(small.num_qubits)
    meas_map: dict[int, int] = {}   # classical bit index -> active-qubit index
    for instr in small.data:
        op = instr.operation
        if op.name == "measure":
            qbit = small.find_bit(instr.qubits[0]).index
            cbit = small.find_bit(instr.clbits[0]).index
            meas_map[cbit] = qbit
        elif op.name == "barrier":
            continue
        else:
            qubits = [small.find_bit(q).index for q in instr.qubits]
            unitary.append(op, qubits)
    sv = Statevector.from_instruction(unitary)
    probs = np.abs(np.asarray(sv.data)) ** 2  # length 2^small.num_qubits
    # Marginalise: for each 4-bit classical register value, sum probs over the
    # 2^(n_active - 4) configurations of the routing-helper qubits.
    out = {format(i, f"0{N}b"): 0.0 for i in range(2 ** N)}
    for idx in range(2 ** small.num_qubits):
        # Build the 4-bit classical register value from the statevector index.
        # Statevector index `idx` is little-endian (bit 0 = qubit 0).
        cl_value = 0
        for cbit, qbit in meas_map.items():
            if (idx >> qbit) & 1:
                cl_value |= (1 << cbit)
        bit = format(cl_value, f"0{N}b")
        out[bit] += float(probs[idx])
    return out, info


def print_distribution(label: str, dist: dict[str, float], info: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"  ISA: depth={info['depth']} total_ops={info['total_ops']} "
          f"2q={info['n2q']}  ops={info['ops']}")
    for bit in sorted(dist.keys()):
        marker = ""
        if bit in SINGLE_EXC:
            marker += "  single-exc"
        if bit == TARGET_BIT:
            marker += "  *TARGET*"
        print(f"  {bit}   {dist[bit]:.6f}{marker}")
    in_sub = sum(dist[b] for b in SINGLE_EXC)
    print(f"  -- in-subspace sum: {in_sub:.6f}")


def latex_row(bit: str, vals: list[float]) -> str:
    return " & ".join([f"\\texttt{{{bit}}}"] + [f"${v:.6f}$" for v in vals]) + r" \\"


def main() -> None:
    backend = QiskitRuntimeService().backend("ibm_kingston")
    print(f"Device transpile target: {backend.name}  "
          f"({backend.num_qubits} qubits, basis={sorted(backend.operation_names)})")
    print(f"(transpile-only, no hardware submission)")
    configs = [
        ("Option A, t=pi/4, one block",          opt_a(pi / 4)),
        ("Option B (r=4), t=pi/4, one block",    opt_b(pi / 4, 4)),
        ("Option A, t=pi/2, one block",          opt_a(pi / 2)),
        ("Option B (r=8), t=pi/2, one block",    opt_b(pi / 2, 8)),
        ("Option A, (pi/4) twice",               opt_a(pi / 4, twice=True)),
        ("Option B (r=4 each), (pi/4) twice",    opt_b(pi / 4, 4, twice=True)),
    ]
    dists = []
    for label, qc in configs:
        d, info = device_distribution(qc, backend)
        print_distribution(label, d, info)
        dists.append(d)

    print("\n" + "=" * 78)
    print("LaTeX rows: bit & A pi/4 & B pi/4 r4 & A pi/2 & B pi/2 r8 & A x2 & B x2 r4")
    print("=" * 78)
    for bit in sorted(dists[0].keys()):
        print(latex_row(bit, [d[bit] for d in dists]))


if __name__ == "__main__":
    main()
