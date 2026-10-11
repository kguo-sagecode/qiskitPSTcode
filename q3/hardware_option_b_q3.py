"""
Run Option B (PauliEvolutionGate + second-order Suzuki-Trotter) on REAL IBM
Quantum hardware for Q_3 (8 qubits, one qubit per vertex), single block at
t = pi/4.

Theory: noiseless outcome is uniform mixing -- each of the 8 single-excitation
states with probability 1/8.
"""

from numpy import pi

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


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
SHOTS = 4096
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


def build_circuit() -> QuantumCircuit:
    H = xy_hamiltonian()
    evo = PauliEvolutionGate(
        H, time=T, synthesis=SuzukiTrotter(order=TROTTER_ORDER, reps=TROTTER_REPS)
    )
    qc = QuantumCircuit(N)
    qc.x(NOT_QUBIT)
    qc.append(evo, range(N))
    qc.measure_all()
    return qc


def main() -> None:
    qc = build_circuit()

    service = QiskitRuntimeService()
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=N)
    print(f"Using backend: {backend.name}  (qubits: {backend.num_qubits})")

    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend, seed_transpiler=0
    )
    isa_qc = pm.run(qc)

    two_q = sum(1 for inst in isa_qc.data if inst.operation.num_qubits == 2)
    print(f"t = {T} (= pi/4)    Trotter: order={TROTTER_ORDER}, reps={TROTTER_REPS}")
    print(f"ISA circuit: depth={isa_qc.depth()}, 2q-gates={two_q}, "
          f"ops={dict(isa_qc.count_ops())}")

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = SHOTS
    job = sampler.run([isa_qc])
    print(f"\nSubmitted. Job ID: {job.job_id()}")
    print(f"  service.job('{job.job_id()}').result()")

    result = job.result()
    counts = result[0].data.meas.get_counts()
    total = sum(counts.values())

    single_exc = {format(1 << k, f"0{N}b") for k in range(N)}
    in_subspace = sum(c for b, c in counts.items() if b in single_exc)

    print(f"\nHardware counts ({total} shots), top 20:")
    for bit, c in sorted(counts.items(), key=lambda kv: -kv[1])[:20]:
        marker = "  <- single exc." if bit in single_exc else ""
        print(f"  |{bit}>: {c:5d}  ({c/total:.4f}){marker}")

    print(f"\nFraction in single-excitation subspace: "
          f"{in_subspace}/{total} = {in_subspace/total:.4f}  (noiseless: 1.0000)")
    print(f"Per single-excitation state, noiseless: 1/{N} = {1/N:.4f}")


if __name__ == "__main__":
    main()
