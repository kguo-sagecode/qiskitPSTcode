"""
Option-B PST hardware runs for the Hamming-weighted paths HP_n,
n = 2, 3, 4 (so 3, 4, 5 qubits), with second-order Suzuki-Trotter at
r = 8 and t = pi/2. These give the HP_n values in the comparison with
PST on Q_n (report, Section 7.3).

Source: vertex 0; target: vertex n (bit-string with bit n set, i.e.
|2^n> in Qiskit little-endian). For each n the script submits a single
job to the least busy available backend (the runs in the report were on
ibm_kingston), waits for results, and prints depth/2q-count, target
probability, in-subspace fraction, and QPU usage (s) from job.usage().

Running this script submits 3 jobs to IBM Quantum hardware and uses QPU
time.
"""

from __future__ import annotations

from math import pi

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from hamming_path_common import option_b_circuit, single_excitation_bitstrings


N_VALUES = [2, 3, 4]
TROTTER_REPS = 8
SHOTS = 4096


def run_one(n: int, service: QiskitRuntimeService) -> None:
    print("\n" + "=" * 78)
    print(f"  HP(Q_{n})  ({n+1} qubits)   Option B, r={TROTTER_REPS}, t=pi/2")
    print("=" * 78)

    qc = option_b_circuit(n, pi / 2, reps=TROTTER_REPS)
    qc.measure_all()

    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=n + 1)
    print(f"  backend: {backend.name}  (qubits: {backend.num_qubits})")

    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend, seed_transpiler=0
    )
    isa = pm.run(qc)
    ops = dict(isa.count_ops())
    two_q = sum(ops.get(k, 0) for k in ("ecr", "cz", "cx", "swap"))
    print(f"  ISA circuit: depth={isa.depth()}, total_ops={sum(ops.values())}, "
          f"2q={two_q},  ops={ops}")

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = SHOTS
    job = sampler.run([isa])
    print(f"  job ID: {job.job_id()}", flush=True)
    print(f"  retrievable via: service.job('{job.job_id()}')", flush=True)

    result = job.result()
    counts = result[0].data.meas.get_counts()
    total = sum(counts.values())

    target_bit = format(1 << n, f"0{n+1}b")
    se = set(single_excitation_bitstrings(n))
    in_sub = sum(c for b, c in counts.items() if b in se) / total
    p_target = counts.get(target_bit, 0) / total
    usage = job.usage()

    print(f"\n  shots: {total}")
    print(f"  in-subspace fraction: {in_sub:.6f}")
    print(f"  P(|{target_bit}>, target = vertex {n}): {p_target:.6f}")
    print(f"  QPU usage: {usage} s")

    print(f"\n  Single-excitation rows ({n+1}):")
    for k in range(n + 1):
        bit = format(1 << k, f"0{n+1}b")
        c = counts.get(bit, 0)
        marker = "  *TARGET*" if k == n else ""
        print(f"    vertex {k}    {bit}    {c}  ({c/total:.6f}){marker}")


def main() -> None:
    service = QiskitRuntimeService()
    for n in N_VALUES:
        run_one(n, service)


if __name__ == "__main__":
    main()
