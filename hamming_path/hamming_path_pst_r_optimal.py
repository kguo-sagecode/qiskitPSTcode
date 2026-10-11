"""
For each Hamming-weighted path HP_n, n = 2..20 (3..21 qubits), find the
Trotter rep count r* in {2, 3, 4, 5, 6, 7, 8, 10, 12, 16} that maximizes
the predicted hardware target probability (report, Section 7.2).

Same method as ../xn/xn_pst_r_optimal.py: for each (n, r) pair, transpile
the Option-B circuit at t = pi/2 and compute:
  - the two-qubit (cz) gate count n_2q,
  - the exact noiseless P(target = vertex n) (bit-string with bit n set),
    from the statevector of the transpiled circuit,
  - the survival fraction S = (1 - 2e-3)^n_2q (median cz error of
    ibm_kingston),
  - the predicted hardware target probability P_noiseless * S.

Prints rows per n and a summary with r* = argmax_r (P_noiseless * S).
"""

from __future__ import annotations

import time
from math import pi

from qiskit import transpile
from qiskit_aer import AerSimulator

from hamming_path_common import option_b_circuit


BASIS = ["cz", "sx", "rz", "x"]            # kingston (Heron r2) native basis
N_VALUES = list(range(2, 21))             # n = 2..20 -> qubit counts 3..21
R_VALUES = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]
P2Q_ERROR = 2.0e-3                         # median cz error of ibm_kingston


def measure(n: int, r: int) -> dict:
    qc = option_b_circuit(n, pi / 2, reps=r)
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
    p_target = float(abs(sv.data[1 << n]) ** 2)

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
          f"{'P_noise':>9}  {'surv':>9}  {'HW_proj':>9}  "
          f"{'tr(s)':>6}  {'aer(s)':>7}",
          flush=True)
    print("-" * 84, flush=True)

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
            print(f"{n:>3}  {r:>3}  {n+1:>3}  {res['ecr']:>5d}  "
                  f"{res['p_target']:>9.6f}  {res['survival']:>9.6f}  "
                  f"{res['hw_proj']:>9.6f}  "
                  f"{res['t_tr']:>6.2f}  {res['t_aer']:>7.2f}",
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
        print(f"{n:>3}  {n+1:>6}  {b['r']:>3}  {b['ecr']:>5d}  "
              f"{b['p_target']:>9.6f}  {b['hw_proj']:>9.6f}")

    # Python-importable dict form for downstream scripts.
    print("\nR_PER_N = {", end="")
    print(", ".join(f"{n}: {best_per_n[n]['r']}" for n in N_VALUES), end="")
    print("}")


if __name__ == "__main__":
    main()
