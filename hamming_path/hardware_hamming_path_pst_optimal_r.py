"""
Hamming-weighted-path PST on ibm_kingston at the Trotter rep count r*(n)
from hamming_path_pst_r_optimal.py (report, Section 7.3).

n = 2..20 (qubit counts 3..21), Option B (PauliEvolutionGate + second-order
Suzuki-Trotter), t = pi/2, 4096 shots per job. Source: vertex 0 (qubit 0);
target: vertex n (qubit n), the bit-string with bit n set.

Prints depth, two-qubit gate count, job ID, in-subspace fraction,
P(target), QPU usage (s), and the probabilities of all single-excitation
states.

Running this script submits 19 jobs to IBM Quantum hardware and uses QPU
time.
"""

from __future__ import annotations

from math import pi

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from hamming_path_common import (
    option_b_circuit,
    single_excitation_bitstrings,
)


# r*(n) from hamming_path_pst_r_optimal.py (p_2q = 2.0e-3).
R_PER_N = {
    2: 3, 3: 4, 4: 4, 5: 5, 6: 5, 7: 6, 8: 7, 9: 7, 10: 8, 11: 8,
    12: 10, 13: 3, 14: 10, 15: 4, 16: 12, 17: 4, 18: 12, 19: 4, 20: 5,
}
SHOTS = 4096
BACKEND_NAME = "ibm_kingston"


def run_one(n: int, r: int, backend) -> None:
    N = n + 1
    print("\n" + "=" * 78)
    print(f"  HP(Q_{n})  ({N} qubits)   Option B, r={r} (r*), t=pi/2")
    print(f"   source = vertex 0 (qubit 0),  target = vertex {n} (qubit {n})")
    print("=" * 78)

    qc = option_b_circuit(n, pi / 2, reps=r)
    qc.measure_all()

    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend, seed_transpiler=0
    )
    isa = pm.run(qc)
    ops = dict(isa.count_ops())
    two_q = sum(ops.get(k, 0) for k in ("ecr", "cz", "cx", "swap"))
    print(f"  ISA circuit: depth={isa.depth()}, total_ops={sum(ops.values())}, 2q={two_q}")

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = SHOTS
    job = sampler.run([isa])
    print(f"  job ID: {job.job_id()}", flush=True)

    result = job.result()
    counts = result[0].data.meas.get_counts()
    total = sum(counts.values())

    tgt_bit = format(1 << n, f"0{N}b")
    se = set(single_excitation_bitstrings(n))
    in_sub = sum(c for b, c in counts.items() if b in se) / total
    p_target = counts.get(tgt_bit, 0) / total
    usage = job.usage()

    print(f"\n  shots: {total}")
    print(f"  in-subspace fraction: {in_sub:.6f}")
    print(f"  P(|{tgt_bit}>, target = vertex {n}): {p_target:.6f}")
    print(f"  QPU usage: {usage} s")

    print(f"\n  Single-excitation rows ({N}):")
    for k in range(N):
        bit = format(1 << k, f"0{N}b")
        c = counts.get(bit, 0)
        marker = ""
        if k == n:
            marker += "  *TARGET*"
        if k == 0:
            marker += "  (source)"
        print(f"    vertex {k:>2}    {bit}    {c}  ({c/total:.6f}){marker}")


def main() -> None:
    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)
    print(f"Pinned backend: {backend.name}  (qubits: {backend.num_qubits})")
    for n in sorted(R_PER_N):
        run_one(n, R_PER_N[n], backend)


if __name__ == "__main__":
    main()
