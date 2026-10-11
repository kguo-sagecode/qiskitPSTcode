"""
For each X_n, n = 2..11 (4, 6, ..., 22 qubits), find the Trotter rep count
r* in {2, 3, 4, 5, 6, 7, 8, 10, 12, 16} that maximizes the predicted
hardware target probability (report, Section 8.3).

For each (n, r) pair, transpile the Option-B circuit at t = pi and
compute:
  - the two-qubit (cz) gate count n_2q,
  - the exact noiseless P(target), from the statevector of the transpiled
    circuit,
  - the survival fraction S = (1 - 2e-3)^n_2q (median cz error of
    ibm_kingston),
  - the predicted hardware target probability P_noiseless * S.

Output rows per n, plus a summary line picking the r that maximizes the
predicted hardware target probability.
"""

from __future__ import annotations

import time
from math import pi

from qiskit import transpile
from qiskit_aer import AerSimulator

from xn_path_common import option_b_circuit, target_qubit


BASIS = ["cz", "sx", "rz", "x"]            # kingston (Heron r2) native basis
N_VALUES = list(range(2, 12))             # n = 2..11 -> qubit counts 4..22
R_VALUES = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]
P2Q_ERROR = 2.0e-3                         # median cz error of ibm_kingston


def measure(n: int, r: int) -> dict:
    qc = option_b_circuit(n, pi, reps=r)
    qc.measure_all()
    t0 = time.time()
    qct = transpile(qc, basis_gates=BASIS, optimization_level=3, seed_transpiler=0)
    t_tr = time.time() - t0
    ops = dict(qct.count_ops())
    ecr = sum(ops.get(k, 0) for k in ("ecr", "cz", "cx", "swap"))

    # Exact noiseless target probability: statevector of the same
    # transpiled circuit, with the final measurements removed.
    qct_sv = qct.remove_final_measurements(inplace=False)
    qct_sv.save_statevector()
    sim = AerSimulator(method="statevector")
    t0 = time.time()
    sv = sim.run(qct_sv).result().get_statevector()
    t_aer = time.time() - t0
    p_target = float(abs(sv.data[1 << target_qubit(n)]) ** 2)

    survival = (1.0 - P2Q_ERROR) ** ecr
    return {
        "ecr": ecr,
        "p_target": p_target,
        "survival": survival,
        "hw_proj": p_target * survival,
        "t_tr": t_tr,
        "t_aer": t_aer,
    }


def main() -> None:
    print(f"{'n':>3}  {'r':>3}  {'q':>3}  {'2q':>5}  "
          f"{'P_noise':>9}  {'surv':>9}  {'HW_proj':>9}  {'aer(s)':>7}",
          flush=True)
    print("-" * 76, flush=True)

    best_per_n = {}
    for n in N_VALUES:
        best = None
        for r in R_VALUES:
            try:
                res = measure(n, r)
            except BaseException as e:
                print(f"{n:>3}  {r:>3}  FAILED ({type(e).__name__}): {e}", flush=True)
                continue
            if best is None or res["hw_proj"] > best["hw_proj"]:
                best = {"r": r, **res}
            print(f"{n:>3}  {r:>3}  {2*n:>3}  {res['ecr']:>5d}  "
                  f"{res['p_target']:>9.6f}  {res['survival']:>9.6f}  "
                  f"{res['hw_proj']:>9.6f}  {res['t_aer']:>7.2f}",
                  flush=True)
        best_per_n[n] = best
        print(f"      ^ best for n={n}: r={best['r']}  "
              f"HW_proj={best['hw_proj']:.6f}  2q={best['ecr']}",
              flush=True)
        print(flush=True)

    print("\nSUMMARY: best r for each n (predicted hardware target probability)")
    print(f"{'n':>3}  {'qubits':>6}  {'r*':>3}  {'2q':>5}  "
          f"{'P_noise':>9}  {'HW_proj':>9}")
    for n in N_VALUES:
        b = best_per_n[n]
        print(f"{n:>3}  {2*n:>6}  {b['r']:>3}  {b['ecr']:>5d}  "
              f"{b['p_target']:>9.6f}  {b['hw_proj']:>9.6f}")


if __name__ == "__main__":
    main()
