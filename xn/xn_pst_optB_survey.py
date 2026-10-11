"""
Option-B gate-count survey for the X_n peak-state-transfer circuits at
t = pi, r = 8, n = 2..20 (report, Section 8.2).

Option A (UnitaryGate + QSD) is covered for n = 2..5 by xn_pst_survey.py
and is impractical for n >= 6 (the 12-qubit transpile has millions of
gates). Option B (PauliEvolutionGate + Suzuki-Trotter) has n_2q = 64n - 62
two-qubit gates and runs up to n = 20.
"""

from __future__ import annotations

import time
from math import pi

from qiskit import transpile

from xn_path_common import option_b_circuit


BASIS = ["cz", "sx", "rz", "x"]
TROTTER_REPS = 8


def main() -> None:
    print(f"{'n':>3}  {'qubits':>6}  {'edges':>5}  "
          f"{'depth':>6}  {'total':>6}  {'2q':>6}  {'time(s)':>8}",
          flush=True)
    print("-" * 60, flush=True)

    for n in range(2, 21):
        qc = option_b_circuit(n, pi, reps=TROTTER_REPS)
        t0 = time.time()
        qct = transpile(qc, basis_gates=BASIS, optimization_level=3, seed_transpiler=0)
        elapsed = time.time() - t0
        ops = dict(qct.count_ops())
        two_q = sum(ops.get(k, 0) for k in ("ecr", "cz", "cx", "swap"))
        print(f"{n:>3}  {2*n:>6}  {2*n - 1:>5}  "
              f"{qct.depth():>6d}  {sum(ops.values()):>6d}  {two_q:>6d}  "
              f"{elapsed:>8.3f}",
              flush=True)


if __name__ == "__main__":
    main()
