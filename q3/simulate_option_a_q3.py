"""
Aer simulation of the Option-A circuit (UnitaryGate + Quantum Shannon
Decomposition) on Q_3 (8 qubits, one qubit per vertex) at t = pi/4.

Noiseless target: uniform mixing on Q_3, i.e. each of the 8 single-excitation
states |q_k = 1> for k = 0, ..., 7 has probability 1/8 = 0.125.
"""

from numpy import pi
from scipy.linalg import expm

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator


N = 8
EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
]
T = pi / 4
NOT_QUBIT = 0
SHOTS = 20000


def xy_hamiltonian() -> SparsePauliOp:
    """H = (1/2) sum_{(i,j) in E(Q_3)} (X_i X_j + Y_i Y_j) in Qiskit little-endian."""
    terms = []
    for i, j in EDGES:
        for P in ("X", "Y"):
            lab = ["I"] * N
            lab[i] = P
            lab[j] = P
            terms.append(("".join(reversed(lab)), 0.5))
    return SparsePauliOp.from_list(terms)


def main() -> None:
    H = xy_hamiltonian()
    U = expm(-1j * T * H.to_matrix())

    qc = QuantumCircuit(N)
    qc.x(NOT_QUBIT)
    qc.append(UnitaryGate(U, label="U_XY"), range(N))
    qc.measure_all()

    basis = ["cz", "sx", "rz", "x"]
    qc_t = transpile(qc, basis_gates=basis, optimization_level=3, seed_transpiler=0)

    two_q = sum(1 for inst in qc_t.data if inst.operation.num_qubits == 2)
    print(f"Transpiled circuit: depth={qc_t.depth()}, 2q={two_q}, "
          f"total_ops={len(qc_t.data)}, op_counts={dict(qc_t.count_ops())}")

    sim = AerSimulator(method="statevector")
    result = sim.run(qc_t, shots=SHOTS, seed_simulator=0).result()
    counts = result.get_counts()

    single_exc = {format(1 << k, f"0{N}b") for k in range(N)}
    in_sub = sum(c for b, c in counts.items() if b in single_exc)

    print(f"\nAer counts ({SHOTS} shots), top 16:")
    for bit, c in sorted(counts.items(), key=lambda kv: -kv[1])[:16]:
        marker = "  <- single excitation" if bit in single_exc else ""
        print(f"  |{bit}>: {c:5d}  ({c/SHOTS:.4f}){marker}")
    print(f"\nIn-subspace: {in_sub}/{SHOTS} = {in_sub/SHOTS:.4f}  (noiseless: 1.0000)")
    print(f"Per single-excitation state, noiseless: 1/{N} = {1/N:.4f}")


if __name__ == "__main__":
    main()
