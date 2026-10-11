"""
Transpilation and noiseless simulation of the Hamming-weighted-path PST
circuits, in the default cz basis and with fractional gates (rzz)
enabled, against the ibm_kingston targets.

For each n = 2..20 at r*(n) (chosen for the cz model in
hamming_path_pst_r_optimal.py and reused for the fractional circuits;
report, Section 7.3):

  - transpile Option B (PauliEvolutionGate, t = pi/2) at opt level 3,
    seed 0, to (a) the cz target and (b) the fractional target followed
    by FoldRzzAngle;
  - record depth / cz / rzz / n_2q for both;
  - compute the exact noiseless P(target) of each device-transpiled
    circuit via Statevector on the active subcircuit, and check the two
    bases agree.

Requires an IBM Quantum account to fetch the backend target; nothing is
submitted. Results: printed table + LaTeX rows + hamming_fractional_survey.json.
"""

from __future__ import annotations

import json
from math import pi

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.transpiler.passes.basis import FoldRzzAngle

from hamming_path_common import option_b_circuit

# r*(n) chosen for the cz survival model
# (p_2q = 2.0e-3); identical to R_PER_N in hardware_hamming_path_pst_optimal_r.py.
R_PER_N = {
    2: 3, 3: 4, 4: 4, 5: 5, 6: 5, 7: 6, 8: 7, 9: 7, 10: 8, 11: 8,
    12: 10, 13: 3, 14: 10, 15: 4, 16: 12, 17: 4, 18: 12, 19: 4, 20: 5,
}
RESULTS_FILE = "hamming_fractional_survey.json"


def fold_and_clean(qct: QuantumCircuit) -> QuantumCircuit:
    folded = PassManager([FoldRzzAngle()]).run(qct)
    out = qct.copy_empty_like()
    for inst in folded.data:
        if inst.operation.name == "global_phase":
            out.global_phase += inst.operation.params[0]
        else:
            out.append(inst)
    return out


def exact_probs(isa: QuantumCircuit, n: int) -> tuple[float, float]:
    """Exact noiseless (P(target), in-subspace sum) of a device-transpiled
    circuit: extract active subcircuit, Statevector, marginalise."""
    active = sorted({isa.find_bit(q).index for ins in isa.data for q in ins.qubits
                     if ins.operation.name != "barrier"})
    ph2new = {p: i for i, p in enumerate(active)}
    uni = QuantumCircuit(len(active))
    meas: dict[int, int] = {}
    for ins in isa.data:
        if ins.operation.name == "measure":
            meas[isa.find_bit(ins.clbits[0]).index] = \
                ph2new[isa.find_bit(ins.qubits[0]).index]
        elif ins.operation.name != "barrier":
            uni.append(ins.operation, [ph2new[isa.find_bit(q).index] for q in ins.qubits])
    probs = np.abs(Statevector.from_instruction(uni).data) ** 2
    qubit_of_clbit = [meas[c] for c in range(n + 1)]
    p_target = 0.0
    p_subspace = 0.0
    target_cl = 1 << n
    for idx, pr in enumerate(probs):
        cl = 0
        for cbit, qbit in enumerate(qubit_of_clbit):
            if (idx >> qbit) & 1:
                cl |= 1 << cbit
        if cl == target_cl:
            p_target += pr
        if cl and (cl & (cl - 1)) == 0:   # exactly one bit set
            p_subspace += pr
    return p_target, p_subspace


def main() -> None:
    service = QiskitRuntimeService()
    king_cz = service.backend("ibm_kingston", use_fractional_gates=False)
    king_fr = service.backend("ibm_kingston", use_fractional_gates=True)
    pm_cz = generate_preset_pass_manager(
        optimization_level=3, backend=king_cz, seed_transpiler=0
    )
    pm_fr = generate_preset_pass_manager(
        optimization_level=3, backend=king_fr, seed_transpiler=0
    )
    print("(transpile + Aer only, no hardware submission)\n")
    hdr = (f"{'n':>3s} {'r*':>3s} {'qb':>3s} "
           f"{'2q':>5s} {'depth_cz':>9s} {'depth_fr':>9s} {'cz_fr':>6s} {'rzz_fr':>7s} "
           f"{'P_target':>10s} {'match':>6s}")
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for n in range(2, 21):
        r = R_PER_N[n]
        qc = option_b_circuit(n, pi / 2, reps=r)
        qc.measure_all()

        isa_cz = pm_cz.run(qc)
        isa_fr = fold_and_clean(pm_fr.run(qc))
        angles = [float(i.operation.params[0]) for i in isa_fr.data
                  if i.operation.name == "rzz"]
        assert all(0 < a <= pi / 2 + 1e-12 for a in angles), f"n={n}: rzz out of range"

        ops_cz = dict(isa_cz.count_ops())
        ops_fr = dict(isa_fr.count_ops())
        n2q_cz = ops_cz.get("cz", 0) + ops_cz.get("rzz", 0)
        n2q_fr = ops_fr.get("cz", 0) + ops_fr.get("rzz", 0)

        p_cz, sub_cz = exact_probs(isa_cz, n)
        p_fr, sub_fr = exact_probs(isa_fr, n)
        match = abs(p_cz - p_fr) < 5e-7

        results[n] = {
            "r": r, "qubits": n + 1,
            "cz_basis": {"depth": isa_cz.depth(), "cz": ops_cz.get("cz", 0),
                         "n2q": n2q_cz, "P_target": p_cz, "in_subspace": sub_cz},
            "fractional": {"depth": isa_fr.depth(), "cz": ops_fr.get("cz", 0),
                           "rzz": ops_fr.get("rzz", 0), "n2q": n2q_fr,
                           "P_target": p_fr, "in_subspace": sub_fr},
        }
        print(f"{n:>3d} {r:>3d} {n+1:>3d} "
              f"{n2q_cz:>5d} {isa_cz.depth():>9d} {isa_fr.depth():>9d} "
              f"{ops_fr.get('cz', 0):>6d} {ops_fr.get('rzz', 0):>7d} "
              f"{p_cz:>10.6f} {'OK' if match else 'DIFF':>6s}")
        if n2q_cz != n2q_fr:
            print(f"    NOTE: n2q differs between bases ({n2q_cz} vs {n2q_fr})")

    with open(RESULTS_FILE, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")

    print("\nLaTeX rows: n & qubits & r* & n2q & depth(cz) & depth(frac) & P_noiseless")
    for n, d in results.items():
        print(f"${n}$ & ${d['qubits']}$ & ${d['r']}$ & ${d['cz_basis']['n2q']}$ & "
              f"${d['cz_basis']['depth']}$ & ${d['fractional']['depth']}$ & "
              f"${d['cz_basis']['P_target']:.6f}$ \\\\")


if __name__ == "__main__":
    main()
