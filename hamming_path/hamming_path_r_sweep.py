"""
Trotter-rep sweep for the Hamming-weighted-path PST circuit at t = pi/2
(report, Section 7.2), for n in {4, 8, 12, 16, 20} and r in
{2, 4, 8, 16, 32}.

For each (n, r) we report:
  - the two-qubit (cz) gate count n_2q after transpilation,
  - the exact noiseless P(target), from the statevector of the transpiled
    circuit,
  - the survival fraction S = (1 - 2e-3) ** n_2q,
  - the predicted hardware target probability P_noiseless * S.
"""

from __future__ import annotations

import time
from math import pi

from qiskit import transpile
from qiskit_aer import AerSimulator

from hamming_path_common import option_b_circuit


BASIS = ["cz", "sx", "rz", "x"]
N_VALUES = [4, 8, 12, 16, 20]
R_VALUES = [2, 4, 8, 16, 32]
P2Q_ERROR = 2.0e-3   # median cz error of ibm_kingston


def measure(n: int, r: int) -> dict:
    qc = option_b_circuit(n, pi / 2, reps=r)
    qc.measure_all()

    t0 = time.time()
    qct = transpile(qc, basis_gates=BASIS, optimization_level=3, seed_transpiler=0)
    transpile_time = time.time() - t0
    ops = dict(qct.count_ops())
    ecr = sum(ops.get(k, 0) for k in ("ecr", "cz", "cx", "swap"))

    # Exact noiseless target probability: statevector of the same
    # transpiled circuit, with the final measurements removed.
    qct_sv = qct.remove_final_measurements(inplace=False)
    qct_sv.save_statevector()
    sim = AerSimulator(method="statevector")
    t0 = time.time()
    sv = sim.run(qct_sv).result().get_statevector()
    aer_time = time.time() - t0
    p_target = float(abs(sv.data[1 << n]) ** 2)

    survival = (1.0 - P2Q_ERROR) ** ecr
    return {
        "ecr": ecr,
        "p_target": p_target,
        "survival": survival,
        "hw_projection": p_target * survival,
        "transpile_time": transpile_time,
        "aer_time": aer_time,
    }


def main() -> None:
    print(f"{'n':>3}  {'r':>3}  {'qubits':>6}  {'2q':>5}  "
          f"{'noiseless':>10}  {'survival':>9}  {'HW proj.':>9}  "
          f"{'transpile(s)':>13}  {'aer(s)':>8}",
          flush=True)
    print("-" * 92, flush=True)

    for n in N_VALUES:
        for r in R_VALUES:
            try:
                res = measure(n, r)
                print(
                    f"{n:>3}  {r:>3}  {n+1:>6}  {res['ecr']:>5d}  "
                    f"{res['p_target']:>10.6f}  {res['survival']:>9.6f}  "
                    f"{res['hw_projection']:>9.6f}  "
                    f"{res['transpile_time']:>13.3f}  {res['aer_time']:>8.3f}",
                    flush=True,
                )
            except BaseException as e:
                print(f"{n:>3}  {r:>3}  FAILED  {type(e).__name__}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
