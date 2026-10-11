"""
Noiseless per-vertex probabilities for X_n, n = 2..11, at the Trotter rep
count r*(n) from xn_pst_r_optimal.py (p_2q = 2.0e-3). Statevector +
20k-shot AerSimulator.

Reports per-n timing: transpile, statevector, shots-sample, total
wall-clock. The statevector values are the Aer columns of the per-vertex
table in Appendix B of the report.
"""

from __future__ import annotations

import time
from math import pi

import numpy as np

from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from xn_path_common import (
    option_b_circuit,
    single_excitation_bitstrings,
    target_qubit,
)


BASIS = ["cz", "sx", "rz", "x"]            # kingston (Heron r2) native basis
SHOTS = 20000

# r*(n) from xn_pst_r_optimal.py (p_2q = 2.0e-3).
R_PER_N = {
    2: 5, 3: 6, 4: 7, 5: 8, 6: 4, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8,
}


def report(n: int, r: int) -> dict:
    print(f"\n=== X_{n}  ({2*n} qubits, target = qubit {target_qubit(n)})  Option B, r={r}, t=pi ===",
          flush=True)
    t0_total = time.time()
    qc_nm = option_b_circuit(n, pi, reps=r)
    qc = qc_nm.copy()
    qc.measure_all()
    t0 = time.time()
    qct_nm = transpile(qc_nm, basis_gates=BASIS, optimization_level=3, seed_transpiler=0)
    qct = transpile(qc, basis_gates=BASIS, optimization_level=3, seed_transpiler=0)
    t_tr = time.time() - t0
    t0 = time.time()
    sv = Statevector.from_instruction(qct_nm)
    t_sv = time.time() - t0
    probs = np.abs(np.asarray(sv.data)) ** 2
    sim = AerSimulator(method="statevector")
    t0 = time.time()
    counts = sim.run(qct, shots=SHOTS, seed_simulator=0).result().get_counts()
    t_sh = time.time() - t0
    t_total = time.time() - t0_total

    n_2q = sum(qct.count_ops().get(g, 0) for g in ("cz", "ecr", "cx"))

    se_bits = single_excitation_bitstrings(n)
    in_sub_sv = sum(probs[int(b, 2)] for b in se_bits)
    in_sub_shots = sum(counts.get(b, 0) for b in se_bits) / SHOTS

    print(f"  n_2q={n_2q}   transpile={t_tr:.2f}s   sv={t_sv:.2f}s   "
          f"shots(20k)={t_sh:.2f}s   total={t_total:.2f}s")
    print(f"  in-subspace (sv): {in_sub_sv:.6f}    (shots): {in_sub_shots:.6f}")
    for k in range(2 * n):
        bit = format(1 << k, f"0{2*n}b")
        p_sv = float(probs[1 << k])
        c = counts.get(bit, 0)
        marker = "  *TARGET*" if k == target_qubit(n) else ""
        print(f"  qubit {k:>2}    {bit}    sv={p_sv:.6f}    shots={c/SHOTS:.6f}{marker}",
              flush=True)
    return {"n": n, "r": r, "n_2q": n_2q, "t_tr": t_tr, "t_sv": t_sv,
            "t_sh": t_sh, "t_total": t_total,
            "p_target": float(probs[1 << target_qubit(n)])}


def main() -> None:
    results = []
    for n in sorted(R_PER_N):
        results.append(report(n, R_PER_N[n]))
    print("\n\n=== TIMING SUMMARY ===")
    print(f"{'n':>3} {'r*':>3} {'n_2q':>5} {'transpile':>10} {'sv':>7} "
          f"{'shots(20k)':>11} {'total':>7} {'P(target)':>10}")
    for r in results:
        print(f"{r['n']:>3} {r['r']:>3} {r['n_2q']:>5} {r['t_tr']:>9.2f}s "
              f"{r['t_sv']:>6.2f}s {r['t_sh']:>10.2f}s "
              f"{r['t_total']:>6.2f}s {r['p_target']:>10.6f}")
    print(f"\ntotal wall-clock: {sum(r['t_total'] for r in results):.2f}s")


if __name__ == "__main__":
    main()
