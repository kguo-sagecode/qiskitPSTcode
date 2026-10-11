"""
Aer simulation of the Option-B circuit (PauliEvolutionGate + second-order
Suzuki-Trotter) on Q_3 (8 qubits, one qubit per vertex) at t = pi/4.

Noiseless target: uniform mixing on Q_3, each single-excitation state at 1/8.
"""

from numpy import pi

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
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
TROTTER_ORDER = 2
TROTTER_REPS = 4


def xy_hamiltonian() -> SparsePauliOp:
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
    evo = PauliEvolutionGate(
        H, time=T, synthesis=SuzukiTrotter(order=TROTTER_ORDER, reps=TROTTER_REPS)
    )

    qc = QuantumCircuit(N)
    qc.x(NOT_QUBIT)
    qc.append(evo, range(N))
    qc.measure_all()

    basis = ["cz", "sx", "rz", "x"]
    qc_t = transpile(qc, basis_gates=basis, optimization_level=3, seed_transpiler=0)

    two_q = sum(1 for inst in qc_t.data if inst.operation.num_qubits == 2)
    print(f"Trotter: order={TROTTER_ORDER}, reps={TROTTER_REPS}, t={T:.6f}")
    print(f"Transpiled circuit: depth={qc_t.depth()}, 2q={two_q}, "
          f"total_ops={len(qc_t.data)}, op_counts={dict(qc_t.count_ops())}")

    sim = AerSimulator(method="statevector")
    counts = sim.run(qc_t, shots=SHOTS, seed_simulator=0).result().get_counts()

    single_exc = {format(1 << k, f"0{N}b") for k in range(N)}
    in_sub = sum(c for b, c in counts.items() if b in single_exc)

    print(f"\nAer counts ({SHOTS} shots), top 16:")
    for bit, c in sorted(counts.items(), key=lambda kv: -kv[1])[:16]:
        marker = "  <- single exc." if bit in single_exc else ""
        print(f"  |{bit}>: {c:5d}  ({c/SHOTS:.4f}){marker}")
    print(f"\nIn-subspace: {in_sub}/{SHOTS} = {in_sub/SHOTS:.4f}  (noiseless: 1.0000)")
    print(f"Per single-excitation state, noiseless: 1/{N} = {1/N:.4f}")


if __name__ == "__main__":
    main()
