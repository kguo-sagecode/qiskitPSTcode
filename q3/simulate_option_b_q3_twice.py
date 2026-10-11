"""
Aer simulation of the Option-B circuit (PauliEvolutionGate + second-order
Suzuki-Trotter) on Q_3 (8 qubits) -- two consecutive blocks of t = pi/4
(= pi/2 total).

Noiseless target: PST from vertex 0 to vertex 7, |10000000>. The total
Trotter work matches one direct r=8 block of pi/2, so the depth / 2q count
should agree with simulate_option_b_q3_pi_over_2.py.
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
T = pi / 4  # per block
NOT_QUBIT = 0
TARGET_QUBIT = 7
SHOTS = 20000
TROTTER_ORDER = 2
TROTTER_REPS = 4  # per block


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
    qc.append(evo, range(N))
    qc.measure_all()

    basis = ["cz", "sx", "rz", "x"]
    qc_t = transpile(qc, basis_gates=basis, optimization_level=3, seed_transpiler=0)

    two_q = sum(1 for inst in qc_t.data if inst.operation.num_qubits == 2)
    print(f"Trotter (per block): order={TROTTER_ORDER}, reps={TROTTER_REPS}, t={T:.6f}")
    print(f"Transpiled circuit: depth={qc_t.depth()}, 2q={two_q}, "
          f"total_ops={len(qc_t.data)}, op_counts={dict(qc_t.count_ops())}")

    sim = AerSimulator(method="statevector")
    counts = sim.run(qc_t, shots=SHOTS, seed_simulator=0).result().get_counts()

    target_bit = format(1 << TARGET_QUBIT, f"0{N}b")
    single_exc = {format(1 << k, f"0{N}b") for k in range(N)}
    in_sub = sum(c for b, c in counts.items() if b in single_exc)
    on_target = counts.get(target_bit, 0)

    print(f"\nAer counts ({SHOTS} shots), top 16:")
    for bit, c in sorted(counts.items(), key=lambda kv: -kv[1])[:16]:
        marker = "  <- single exc." if bit in single_exc else ""
        if bit == target_bit:
            marker += "  *TARGET*"
        print(f"  |{bit}>: {c:5d}  ({c/SHOTS:.4f}){marker}")
    print(f"\nIn-subspace: {in_sub}/{SHOTS} = {in_sub/SHOTS:.4f}  (noiseless: 1.0000)")
    print(f"On target |{target_bit}>: {on_target}/{SHOTS} = {on_target/SHOTS:.4f}  (noiseless: 1.0000)")


if __name__ == "__main__":
    main()
