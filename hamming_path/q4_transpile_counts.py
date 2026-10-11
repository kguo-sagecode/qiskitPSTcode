"""
Estimate the transpilation cost of PST on the 4-cube Q_4 (16 qubits, one
per vertex) on a Heron-family IBM backend, for Option B (Trotter,
PauliEvolutionGate) at several Trotter rep counts and times. The report
(Section 7.3) quotes the two-qubit gate count at t = pi/2 with r = 4.

Option A (UnitaryGate / Quantum Shannon Decomposition) is infeasible at
16 qubits (it needs a 65,536 x 65,536 dense matrix, about 64 GB), so only
Option B is reported.

Uses a fake Heron-family backend, so this runs locally without IBM
credentials. qiskit-ibm-runtime 0.46.1 has no FakeKingston, so the script
falls back to FakeFez (Heron r2, the same basis and heavy-hex layout).
"""

from __future__ import annotations

from math import pi

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


N = 16
EDGES = [
    (0, 1), (0, 2), (0, 4), (0, 8),
    (1, 3), (1, 5), (1, 9),
    (2, 3), (2, 6), (2, 10),
    (3, 7), (3, 11),
    (4, 5), (4, 6), (4, 12),
    (5, 7), (5, 13),
    (6, 7), (6, 14),
    (7, 15),
    (8, 9), (8, 10), (8, 12),
    (9, 11), (9, 13),
    (10, 11), (10, 14),
    (11, 15),
    (12, 13), (12, 14),
    (13, 15),
    (14, 15),
]
NOT_QUBIT = 0


def xy_hamiltonian() -> SparsePauliOp:
    terms = []
    for i, j in EDGES:
        for P in ("X", "Y"):
            lab = ["I"] * N
            lab[i] = P
            lab[j] = P
            terms.append(("".join(reversed(lab)), 0.5))
    return SparsePauliOp.from_list(terms)


def build_one_block(t: float, reps: int) -> QuantumCircuit:
    evo = PauliEvolutionGate(
        xy_hamiltonian(), time=t, synthesis=SuzukiTrotter(order=2, reps=reps)
    )
    qc = QuantumCircuit(N)
    qc.x(NOT_QUBIT)
    qc.append(evo, range(N))
    qc.measure_all()
    return qc


def build_two_blocks(t_each: float, reps_each: int) -> QuantumCircuit:
    """Two back-to-back blocks of evolution time t_each, reps_each Trotter reps each."""
    evo = PauliEvolutionGate(
        xy_hamiltonian(), time=t_each, synthesis=SuzukiTrotter(order=2, reps=reps_each)
    )
    qc = QuantumCircuit(N)
    qc.x(NOT_QUBIT)
    qc.append(evo, range(N))
    qc.append(evo, range(N))
    qc.measure_all()
    return qc


def pick_heron_backend():
    """The available Heron-family fake backends, in order of preference."""
    from qiskit_ibm_runtime import fake_provider as fp

    candidates = []
    for name in ("FakeKingston", "FakeFez", "FakeMarrakesh", "FakeTorino"):
        cls = getattr(fp, name, None)
        if cls is None:
            cls = getattr(fp, name + "V2", None)
        if cls is not None:
            candidates.append((name, cls()))
    return candidates


def report(label: str, qc: QuantumCircuit, backend) -> dict:
    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend, seed_transpiler=0
    )
    isa = pm.run(qc)
    op_counts = dict(isa.count_ops())
    two_q = sum(
        c for op, c in op_counts.items() if op in ("ecr", "cx", "cz", "swap")
    )
    print(
        f"{label:>40s} | depth={isa.depth():4d}  total={sum(op_counts.values()):5d}  "
        f"2q={two_q:4d}  ops={op_counts}"
    )
    return {"label": label, "depth": isa.depth(), "total": sum(op_counts.values()),
            "two_q": two_q, "ops": op_counts}


def main() -> None:
    print("=" * 90)
    print("Q_4 (16 qubits) Option B (Trotter) transpilation cost on Heron-class fakes")
    print("=" * 90)

    backends = pick_heron_backend()
    if not backends:
        raise SystemExit("No Heron fake backends available in this qiskit-ibm-runtime.")

    for name, backend in backends:
        print(f"\nBackend: {name}  (qubits: {backend.num_qubits}, basis: {sorted(backend.operation_names)})")
        rows = []
        rows.append(report("t=pi/4 one block, reps=2",
                           build_one_block(pi / 4, 2), backend))
        rows.append(report("t=pi/4 one block, reps=4",
                           build_one_block(pi / 4, 4), backend))
        rows.append(report("t=pi/2 one block, reps=4",
                           build_one_block(pi / 2, 4), backend))
        rows.append(report("t=pi/2 one block, reps=8",
                           build_one_block(pi / 2, 8), backend))
        rows.append(report("t=pi/4 twice, reps=2 each (= pi/2)",
                           build_two_blocks(pi / 4, 2), backend))
        rows.append(report("t=pi/4 twice, reps=4 each (= pi/2)",
                           build_two_blocks(pi / 4, 4), backend))
        # Report only the first available backend.
        break


if __name__ == "__main__":
    main()
