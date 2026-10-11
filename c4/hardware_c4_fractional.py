"""
The six C_4 hardware circuits of Section 5 of the report, run on
ibm_kingston with fractional gates enabled (native rzz and rx).

Circuits (the same as in the cz-basis scripts hardware_option_*.py;
4096 shots each, one job per circuit so that job.usage() is reported per
circuit):
  1. Option A (UnitaryGate),        t = pi/2          PST
  2. Option B (r=8),                t = pi/2          PST
  3. Option A (UnitaryGate),        t = pi/4          uniform mixing
  4. Option B (r=4),                t = pi/4          uniform mixing
  5. Option A (UnitaryGate) x 2,    t = pi/4 each     PST via mixing
  6. Option B (r=4) x 2,            t = pi/4 each     PST via mixing

Transpilation: optimization_level=3, seed 0, against the fractional
target, followed by FoldRzzAngle to bring all rzz angles into the range
[0, pi/2] accepted by the hardware.

Running this script submits 6 jobs to IBM Quantum hardware and uses QPU
time. Results are written to c4_fractional_hw_results.json.
"""

from __future__ import annotations

import json
from math import pi

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, UnitaryGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.transpiler.passes.basis import FoldRzzAngle
from scipy.linalg import expm


N = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
NOT_QUBIT = 0
SHOTS = 4096
RESULTS_FILE = "c4_fractional_hw_results.json"


def hamiltonian() -> SparsePauliOp:
    terms = []
    for i, j in EDGES:
        for P in ("X", "Y"):
            lab = ["I"] * N
            lab[i] = P
            lab[j] = P
            terms.append(("".join(reversed(lab)), 0.5))
    return SparsePauliOp.from_list(terms)


def opt_a(t: float, blocks: int = 1) -> QuantumCircuit:
    U = expm(-1j * t * hamiltonian().to_matrix())
    qc = QuantumCircuit(N)
    qc.x(NOT_QUBIT)
    for _ in range(blocks):
        qc.append(UnitaryGate(U, label="U"), range(N))
    qc.measure_all()
    return qc


def opt_b(t: float, reps: int, blocks: int = 1) -> QuantumCircuit:
    evo = PauliEvolutionGate(
        hamiltonian(), time=t, synthesis=SuzukiTrotter(order=2, reps=reps)
    )
    qc = QuantumCircuit(N)
    qc.x(NOT_QUBIT)
    for _ in range(blocks):
        qc.append(evo, range(N))
    qc.measure_all()
    return qc


def fold_and_clean(qct: QuantumCircuit) -> QuantumCircuit:
    """Wrap rzz angles into [0, pi/2]; absorb the fold pass's leftover
    GlobalPhaseGate instructions into the circuit's global phase."""
    folded = PassManager([FoldRzzAngle()]).run(qct)
    out = folded.copy_empty_like()
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
        ("A_pi2_pst",        opt_a(pi / 2)),
        ("B8_pi2_pst",       opt_b(pi / 2, 8)),
        ("A_pi4_mixing",     opt_a(pi / 4)),
        ("B4_pi4_mixing",    opt_b(pi / 4, 4)),
        ("A_pi4x2_pst",      opt_a(pi / 4, 2)),
        ("B4_pi4x2_pst",     opt_b(pi / 4, 4, 2)),
    ]

    jobs = []
    for label, qc in cases:
        isa = fold_and_clean(pm.run(qc))
        ops = dict(isa.count_ops())
        angles = [
            float(inst.operation.params[0])
            for inst in isa.data
            if inst.operation.name == "rzz"
        ]
        assert all(0 <= a <= pi / 2 + 1e-12 for a in angles), f"{label}: rzz out of range"
        two_q = ops.get("cz", 0) + ops.get("rzz", 0)
        print(f"{label:<16s} depth={isa.depth():>5d}  cz={ops.get('cz', 0):>4d}  "
              f"rzz={ops.get('rzz', 0):>4d}  2q={two_q:>4d}")

        sampler = Sampler(mode=backend)
        sampler.options.default_shots = SHOTS
        job = sampler.run([isa])
        print(f"  -> submitted, job id: {job.job_id()}")
        jobs.append((label, isa, job))

    print("\nWaiting for results...")
    try:
        with open(RESULTS_FILE) as fh:
            results = json.load(fh)   # merge with an existing results file
    except FileNotFoundError:
        results = {}
    for label, isa, job in jobs:
        res = job.result()
        counts = res[0].data.meas.get_counts()
        usage = job.usage()
        total = sum(counts.values())
        ops = dict(isa.count_ops())
        results[label] = {
            "job_id": job.job_id(),
            "usage_seconds": usage,
            "shots": total,
            "ops": ops,
            "two_q": ops.get("cz", 0) + ops.get("rzz", 0),
            "depth": isa.depth(),
            "counts": counts,
        }
        target = counts.get("0100", 0) / total
        sub = sum(counts.get(b, 0) for b in ("0001", "0010", "0100", "1000")) / total
        print(f"\n{label}:  QPU usage {usage} s")
        print(f"  P(0100) = {target:.6f}   in-subspace = {sub:.6f}")
        for bit, c in sorted(counts.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    |{bit}>: {c:5d}  ({c/total:.6f})")

    with open(RESULTS_FILE, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved all results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
