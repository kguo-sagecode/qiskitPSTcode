"""
The X_n peak-state-transfer sweep on ibm_kingston with fractional gates
enabled: n = 2..11, Option B (PauliEvolutionGate, t = pi) at the r*(n) of
xn_pst_r_optimal.py (not re-optimized for the fractional basis, as for
the Hamming paths), 4096 shots, one job per n.

The cz-basis sweep is hardware_xn_pst_optimal_r.py.

Running this script submits 10 jobs to IBM Quantum hardware and uses QPU
time. Results are written to xn_fractional_hw_results.json.
"""

from __future__ import annotations

import json
from math import pi

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.transpiler.passes.basis import FoldRzzAngle

from xn_path_common import (
    option_b_circuit,
    single_excitation_bitstrings,
    target_bitstring,
)

R_PER_N = {2: 5, 3: 6, 4: 7, 5: 8, 6: 4, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8}
T = pi
SHOTS = 4096
BACKEND_NAME = "ibm_kingston"
RESULTS_FILE = "xn_fractional_hw_results.json"


def fold_and_clean(qct: QuantumCircuit) -> QuantumCircuit:
    folded = PassManager([FoldRzzAngle()]).run(qct)
    out = qct.copy_empty_like()
    for inst in folded.data:
        if inst.operation.name == "global_phase":
            out.global_phase += inst.operation.params[0]
        else:
            out.append(inst)
    return out


def main() -> None:
    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME, use_fractional_gates=True)
    print(f"Backend: {backend.name} (fractional gates enabled)")
    assert "rzz" in backend.operation_names
    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend, seed_transpiler=0
    )

    jobs = []
    for n in sorted(R_PER_N):
        r = R_PER_N[n]
        qc = option_b_circuit(n, T, reps=r)
        qc.measure_all()
        isa = fold_and_clean(pm.run(qc))
        angles = [float(i.operation.params[0]) for i in isa.data
                  if i.operation.name == "rzz"]
        assert all(0 < a <= pi / 2 + 1e-12 for a in angles), f"n={n}: rzz out of range"
        ops = dict(isa.count_ops())
        n2q = ops.get("cz", 0) + ops.get("rzz", 0)
        print(f"X_{n:<2d} (r*={r}): depth={isa.depth():>5d}  cz={ops.get('cz', 0):>3d}  "
              f"rzz={ops.get('rzz', 0):>4d}  2q={n2q:>4d}")
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = SHOTS
        job = sampler.run([isa])
        print(f"   -> submitted, job id: {job.job_id()}")
        jobs.append((n, r, isa, job))

    print("\nWaiting for results...")
    results = {}
    for n, r, isa, job in jobs:
        res = job.result()
        counts = res[0].data.meas.get_counts()
        total = sum(counts.values())
        ops = dict(isa.count_ops())
        target = target_bitstring(n)
        p_target = counts.get(target, 0) / total
        in_sub = sum(counts.get(b, 0) for b in single_excitation_bitstrings(n)) / total
        results[n] = {
            "r": r,
            "job_id": job.job_id(),
            "usage_seconds": job.usage(),
            "shots": total,
            "ops": ops,
            "n2q": ops.get("cz", 0) + ops.get("rzz", 0),
            "depth": isa.depth(),
            "P_target": p_target,
            "in_subspace": in_sub,
            "counts": counts,
        }
        print(f"X_{n:<2d}: P(target)={p_target:.6f}  in-subspace={in_sub:.6f}  "
              f"usage={job.usage()} s")

    with open(RESULTS_FILE, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved all results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
