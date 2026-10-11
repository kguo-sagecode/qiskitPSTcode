"""
X_n peak-state-transfer runs on ibm_kingston at the Trotter rep count
r*(n) from xn_pst_r_optimal.py (report, Section 8.4).

For each (n, r) submits one 4096-shot job and prints the probabilities of
all single-excitation states.

Running this script submits 10 jobs to IBM Quantum hardware and uses QPU
time.
"""

from __future__ import annotations

from math import pi

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from xn_path_common import (
    option_b_circuit,
    single_excitation_bitstrings,
    source_qubit,
    target_qubit,
)


# r*(n) from xn_pst_r_optimal.py (p_2q = 2.0e-3).
R_PER_N = {
    2: 5,
    3: 6,
    4: 7,
    5: 8,
    6: 4,
    7: 4,
    8: 5,
    9: 6,
    10: 7,
    11: 8,
}

SHOTS = 4096
BACKEND_NAME = "ibm_kingston"


def run_one(n: int, r: int, backend) -> None:
    print("\n" + "=" * 78)
    print(f"  X_{n}  ({2*n} qubits)   Option B, r={r} (r*), t=pi")
    print(f"   source = qubit {source_qubit(n)} (w_1),  target = qubit {target_qubit(n)} (w_{n})")
    print("=" * 78)

    qc = option_b_circuit(n, pi, reps=r)
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

    tgt_bit = format(1 << target_qubit(n), f"0{2*n}b")
    se = set(single_excitation_bitstrings(n))
    in_sub = sum(c for b, c in counts.items() if b in se) / total
    p_target = counts.get(tgt_bit, 0) / total
    usage = job.usage()

    print(f"\n  shots: {total}")
    print(f"  in-subspace fraction: {in_sub:.6f}")
    print(f"  P(|{tgt_bit}>, target = qubit {target_qubit(n)}): {p_target:.6f}")
    print(f"  QPU usage: {usage} s")

    print(f"\n  Single-excitation rows ({2*n}):")
    for k in range(2 * n):
        bit = format(1 << k, f"0{2*n}b")
        c = counts.get(bit, 0)
        marker = "  *TARGET*" if k == target_qubit(n) else ""
        if k == source_qubit(n):
            marker += "  (source)"
        print(f"    qubit {k:>2}    {bit}    {c}  ({c/total:.6f}){marker}")


def main() -> None:
    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)
    print(f"Pinned backend: {backend.name}  (qubits: {backend.num_qubits})")
    for n in sorted(R_PER_N):
        run_one(n, R_PER_N[n], backend)


if __name__ == "__main__":
    main()
