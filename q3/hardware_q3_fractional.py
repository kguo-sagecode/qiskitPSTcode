"""
The two Q_3 hardware circuits (8 qubits, one per vertex) run on
ibm_kingston with fractional gates enabled:

  1. PauliEvolutionGate r=8, t = pi/2   (PST, target |e_7> = |10000000>)
  2. PauliEvolutionGate r=4, t = pi/4   (uniform mixing)

4096 shots, one job per circuit. The cz-basis runs are
hardware_option_b_q3_pi_over_2.py and hardware_option_b_q3.py; the report
compares the two bases in Section 6.

Running this script submits 2 jobs to IBM Quantum hardware and uses QPU
time. Results are written to q3_fractional_hw_results.json.
"""

from __future__ import annotations

import json
from math import pi

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.transpiler.passes.basis import FoldRzzAngle

N = 8
EDGES = [(k, k ^ (1 << b)) for k in range(8) for b in range(3) if k < (k ^ (1 << b))]
SHOTS = 4096
RESULTS_FILE = "q3_fractional_hw_results.json"
SINGLE_EXC = [format(1 << k, f"0{N}b") for k in range(N)]
TARGET = format(1 << 7, f"0{N}b")


def hamiltonian() -> SparsePauliOp:
    terms = []
    for i, j in EDGES:
        for P in ("X", "Y"):
            lab = ["I"] * N
            lab[i] = P
            lab[j] = P
            terms.append(("".join(reversed(lab)), 0.5))
    return SparsePauliOp.from_list(terms)


def opt_b(t: float, reps: int) -> QuantumCircuit:
    evo = PauliEvolutionGate(
        hamiltonian(), time=t, synthesis=SuzukiTrotter(order=2, reps=reps)
    )
    qc = QuantumCircuit(N)
    qc.x(0)
    qc.append(evo, range(N))
    qc.measure_all()
    return qc


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
    backend = service.backend("ibm_kingston", use_fractional_gates=True)
    print(f"Backend: {backend.name} (fractional gates enabled)")
    assert "rzz" in backend.operation_names
    pm = generate_preset_pass_manager(
        optimization_level=3, backend=backend, seed_transpiler=0
    )

    cases = [
        ("q3_B8_pi2_pst",    opt_b(pi / 2, 8)),
        ("q3_B4_pi4_mixing", opt_b(pi / 4, 4)),
    ]
    jobs = []
    for label, qc in cases:
        isa = fold_and_clean(pm.run(qc))
        angles = [float(i.operation.params[0]) for i in isa.data
                  if i.operation.name == "rzz"]
        assert all(0 < a <= pi / 2 + 1e-12 for a in angles), f"{label}: rzz out of range"
        ops = dict(isa.count_ops())
        n2q = ops.get("cz", 0) + ops.get("rzz", 0)
        print(f"{label:<18s} depth={isa.depth():>5d}  cz={ops.get('cz', 0):>4d}  "
              f"rzz={ops.get('rzz', 0):>4d}  2q={n2q:>4d}")
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = SHOTS
        job = sampler.run([isa])
        print(f"  -> submitted, job id: {job.job_id()}")
        jobs.append((label, isa, job))

    print("\nWaiting for results...")
    results = {}
    for label, isa, job in jobs:
        res = job.result()
        counts = res[0].data.meas.get_counts()
        total = sum(counts.values())
        ops = dict(isa.count_ops())
        p_target = counts.get(TARGET, 0) / total
        in_sub = sum(counts.get(b, 0) for b in SINGLE_EXC) / total
        per_vertex = {k: counts.get(format(1 << k, f"0{N}b"), 0) / total
                      for k in range(N)}
        results[label] = {
            "job_id": job.job_id(),
            "usage_seconds": job.usage(),
            "shots": total,
            "ops": ops,
            "n2q": ops.get("cz", 0) + ops.get("rzz", 0),
            "depth": isa.depth(),
            "P_target": p_target,
            "in_subspace": in_sub,
            "per_vertex": per_vertex,
            "counts": counts,
        }
        print(f"\n{label}:  QPU usage {job.usage()} s")
        print(f"  P({TARGET}) = {p_target:.6f}   in-subspace = {in_sub:.6f}")
        for k in range(N):
            print(f"    vertex {k}: {per_vertex[k]:.6f}")

    with open(RESULTS_FILE, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved all results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
