"""
The Hamming-weighted-path PST sweep on ibm_kingston with fractional gates
enabled: n = 2..20, Option B (PauliEvolutionGate, t = pi/2) at the r*(n)
of hamming_path_pst_r_optimal.py (not re-optimized for the fractional
basis; report, Section 7.3), 4096 shots, one job per n so that job.usage()
is reported per circuit.

The cz-basis sweep is hardware_hamming_path_pst_optimal_r.py.

Running this script submits 19 jobs to IBM Quantum hardware and uses QPU
time. Results are written to hamming_fractional_hw_results.json.
"""

from __future__ import annotations

import json
from math import pi

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.transpiler.passes.basis import FoldRzzAngle

from hamming_path_common import option_b_circuit

R_PER_N = {
    2: 3, 3: 4, 4: 4, 5: 5, 6: 5, 7: 6, 8: 7, 9: 7, 10: 8, 11: 8,
    12: 10, 13: 3, 14: 10, 15: 4, 16: 12, 17: 4, 18: 12, 19: 4, 20: 5,
}
SHOTS = 4096
BACKEND_NAME = "ibm_kingston"
RESULTS_FILE = "hamming_fractional_hw_results.json"


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
    for n in range(2, 21):
        r = R_PER_N[n]
        qc = option_b_circuit(n, pi / 2, reps=r)
        qc.measure_all()
        isa = fold_and_clean(pm.run(qc))
        angles = [float(i.operation.params[0]) for i in isa.data
                  if i.operation.name == "rzz"]
        assert all(0 < a <= pi / 2 + 1e-12 for a in angles), f"n={n}: rzz out of range"
        ops = dict(isa.count_ops())
        n2q = ops.get("cz", 0) + ops.get("rzz", 0)
        print(f"n={n:>2d} (r*={r:>2d}): depth={isa.depth():>5d}  cz={ops.get('cz', 0):>3d}  "
              f"rzz={ops.get('rzz', 0):>4d}  2q={n2q:>4d}")
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = SHOTS
        job = sampler.run([isa])
        print(f"   -> submitted, job id: {job.job_id()}")
        jobs.append((n, r, isa, job))

    print("\nWaiting for results...")
    try:
        with open(RESULTS_FILE) as fh:
            results = json.load(fh)   # merge with an existing results file
    except FileNotFoundError:
        results = {}
    for n, r, isa, job in jobs:
        res = job.result()
        counts = res[0].data.meas.get_counts()
        total = sum(counts.values())
        ops = dict(isa.count_ops())
        target = "1" + "0" * n
        single_exc = ["0" * (n - k) + "1" + "0" * k for k in range(n + 1)]
        p_target = counts.get(target, 0) / total
        in_sub = sum(counts.get(b, 0) for b in single_exc) / total
        results[str(n)] = {   # str keys, as in a loaded JSON file
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
        print(f"n={n:>2d}: P(target)={p_target:.6f}  in-subspace={in_sub:.6f}  "
              f"usage={job.usage()} s")

    with open(RESULTS_FILE, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved all results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
