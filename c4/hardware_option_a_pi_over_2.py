"""
Run Option A (exact UnitaryGate) on REAL IBM Quantum hardware, single block, t = pi/2.

Circuit: X on q0, then exp(-i (pi/2) H_XY) on the 4-cycle.
Theory: noiseless outcome is |0100> with probability 1 (excitation lands antipodally).
"""

from numpy import pi
from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


N = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
T = pi / 2
NOT_QUBIT = 0
SHOTS = 4096


def xy_hamiltonian_4cycle() -> SparsePauliOp:
    terms = []
    for i, j in EDGES:
        for P in ("X", "Y"):
            lab = ["I"] * N
            lab[i] = P
            lab[j] = P
            terms.append(("".join(reversed(lab)), 0.5))
    return SparsePauliOp.from_list(terms)


def build_circuit() -> QuantumCircuit:
    H = xy_hamiltonian_4cycle()
    U_xy = expm(-1j * T * H.to_matrix())
    qc = QuantumCircuit(N)
    qc.x(NOT_QUBIT)
    qc.append(UnitaryGate(U_xy, label="exp(-itH_XY)"), range(N))
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
    print(f"t = {T}  ( = pi/2 )")
    print(f"ISA circuit: depth={isa_qc.depth()}, 2q-gates={two_q}, "
          f"ops={dict(isa_qc.count_ops())}")
    print(f"Physical qubits used: {[isa_qc.layout.final_index_layout()[i] for i in range(N)]}")

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = SHOTS
    job = sampler.run([isa_qc])
    print(f"\nSubmitted. Job ID: {job.job_id()}")
    print("Waiting for result... (Ctrl-C is safe; retrieve later with the job ID below.)")
    print(f"  service.job('{job.job_id()}').result()")

    result = job.result()
    counts = result[0].data.meas.get_counts()
    total = sum(counts.values())

    print(f"\nHardware counts ({total} shots):")
    expected_states = {"0001", "0010", "0100", "1000"}
    in_subspace = 0
    on_target = counts.get("0100", 0)
    for bit, c in sorted(counts.items(), key=lambda kv: -kv[1])[:16]:
        marker = "  <- single excitation" if bit in expected_states else ""
        marker += "  *TARGET*" if bit == "0100" else ""
        print(f"  |{bit}>: {c:5d}  ({c/total:.4f}){marker}")
        if bit in expected_states:
            in_subspace += c
    print(f"\nFraction in single-excitation subspace: "
          f"{in_subspace}/{total} = {in_subspace/total:.4f}  (noiseless: 1.0000)")
    print(f"Probability on |0100> (target):           "
          f"{on_target/total:.4f}  (noiseless: 1.0000)")


if __name__ == "__main__":
    main()
