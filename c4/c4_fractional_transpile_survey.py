"""
Transpile-only survey of the six C_4 hardware circuits of Section 5 of the
report, transpiled for ibm_kingston with and without fractional gates
(use_fractional_gates=True adds native rzz and rx to the target).

For each configuration we report depth, total gates and the two-qubit gate
count (cz and rzz) in both bases, a noiseless Aer check of the
fractional-basis circuit (target-state probability), and the range of rzz
angles (the hardware requires 0 <= theta <= pi/2).

Requires an IBM Quantum account to fetch the backend target; nothing is
submitted.
"""

from __future__ import annotations

from math import pi

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, UnitaryGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.transpiler.passes.basis import FoldRzzAngle
from scipy.linalg import expm


N = 4
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
NOT_QUBIT = 0
AER_SHOTS = 100_000


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
    """Wrap rzz angles into [0, pi/2] (hardware requirement); the fold pass
    leaves GlobalPhaseGate instructions behind, which we absorb into the
    circuit's global phase so that Aer (and the ISA check) are happy."""
    folded = PassManager([FoldRzzAngle()]).run(qct)
    out = folded.copy_empty_like()
    for inst in folded.data:
        if inst.operation.name == "global_phase":
            out.global_phase += inst.operation.params[0]
        else:
            out.append(inst)
    return out


def stats(qct: QuantumCircuit) -> dict:
    ops = dict(qct.count_ops())
    two_q = sum(ops.get(k, 0) for k in ("ecr", "cz", "cx", "swap", "rzz"))
    rzz_angles = [
        float(inst.operation.params[0])
        for inst in qct.data
        if inst.operation.name == "rzz"
    ]
    return {
        "depth": qct.depth(),
        "total": sum(ops.values()),
        "two_q": two_q,
        "ops": ops,
        "rzz_angles": rzz_angles,
    }


def aer_distribution(qct: QuantumCircuit) -> dict[str, float]:
    """Noiseless sample of a device-transpiled circuit (Aer truncates idle qubits)."""
    sim = AerSimulator()
    counts = sim.run(qct, shots=AER_SHOTS, seed_simulator=0).result().get_counts()
    return {b: c / AER_SHOTS for b, c in counts.items()}


def main() -> None:
    service = QiskitRuntimeService()
    king_cz = service.backend("ibm_kingston", use_fractional_gates=False)
    king_fr = service.backend("ibm_kingston", use_fractional_gates=True)
    print(f"cz target ops:         {sorted(king_cz.operation_names)}")
    print(f"fractional target ops: {sorted(king_fr.operation_names)}")
    print("(transpile-only, no hardware submission)\n")

    pm_cz = generate_preset_pass_manager(
        optimization_level=3, backend=king_cz, seed_transpiler=0
    )
    pm_fr = generate_preset_pass_manager(
        optimization_level=3, backend=king_fr, seed_transpiler=0
    )

    # The six hardware configurations, with their noiseless targets.
    cases = [
        ("A,  t=pi/2 (PST)",          opt_a(pi / 2),        {"0100": 1.0}),
        ("B8, t=pi/2 (PST)",          opt_b(pi / 2, 8),     {"0100": 1.0}),
        ("A,  t=pi/4 (mixing)",       opt_a(pi / 4),        {b: 0.25 for b in ("0001", "0010", "0100", "1000")}),
        ("B4, t=pi/4 (mixing)",       opt_b(pi / 4, 4),     {b: 0.25 for b in ("0001", "0010", "0100", "1000")}),
        ("A,  (pi/4)x2 (PST via UM)", opt_a(pi / 4, 2),     {"0100": 1.0}),
        ("B4, (pi/4)x2 (PST via UM)", opt_b(pi / 4, 4, 2),  {"0100": 1.0}),
    ]

    hdr = (
        f"{'Configuration':<28s}"
        f"{'depth':>6s}{'2q(cz)':>8s}   "
        f"{'depth':>6s}{'cz':>5s}{'rzz':>5s}{'2q':>5s}"
        f"{'rzz angle range':>22s}{'max |dP| vs theory':>20s}"
    )
    print(f"{'':28s}{'-- cz basis --':>14s}   {'------- fractional basis -------':>43s}")
    print(hdr)
    print("-" * len(hdr))

    for label, qc, theory in cases:
        cz_s = stats(pm_cz.run(qc))
        qct_fr = fold_and_clean(pm_fr.run(qc))
        fr_s = stats(qct_fr)
        dist = aer_distribution(qct_fr)
        keys = set(theory) | set(dist)
        max_dev = max(abs(dist.get(k, 0.0) - theory.get(k, 0.0)) for k in keys)
        if fr_s["rzz_angles"]:
            lo, hi = min(fr_s["rzz_angles"]), max(fr_s["rzz_angles"])
            in_range = all(0 <= a <= pi / 2 + 1e-12 for a in fr_s["rzz_angles"])
            ang = f"[{lo:+.3f},{hi:+.3f}]" + ("" if in_range else "  OUT OF RANGE")
        else:
            ang = "(none)"
        print(
            f"{label:<28s}"
            f"{cz_s['depth']:>6d}{cz_s['two_q']:>8d}   "
            f"{fr_s['depth']:>6d}{fr_s['ops'].get('cz', 0):>5d}"
            f"{fr_s['ops'].get('rzz', 0):>5d}{fr_s['two_q']:>5d}"
            f"{ang:>22s}{max_dev:>20.6f}"
        )

    print(
        "\nmax |dP| vs theory is the largest deviation of the noiseless Aer\n"
        "distribution of the fractional-basis circuit from the theoretical one\n"
        f"(sampling noise at {AER_SHOTS} shots ~ 0.003; Option B also carries\n"
        "Trotter error)."
    )


if __name__ == "__main__":
    main()
