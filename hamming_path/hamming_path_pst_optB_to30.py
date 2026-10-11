"""
Extend the Option-B PST survey of Hamming-weighted paths to n = 2..30.

PauliEvolutionGate at r=8, transpile to {cz, sx, rz, x},
optimization_level=3, no coupling map.
"""

from __future__ import annotations

import time
from math import pi

from qiskit import transpile

from hamming_path_common import option_b_circuit


BASIS = ["cz", "sx", "rz", "x"]
TROTTER_REPS = 8


def main() -> None:
    print(f"{'n':>3}  {'qubits':>6}  {'edges':>5}  "
          f"{'depth':>6}  {'total':>6}  {'2q':>6}  {'time(s)':>8}",
          flush=True)
    print("-" * 60, flush=True)

    for n in range(2, 31):
        try:
            qc = option_b_circuit(n, pi / 2, reps=TROTTER_REPS)
            t0 = time.time()
            qct = transpile(qc, basis_gates=BASIS, optimization_level=3, seed_transpiler=0)
            elapsed = time.time() - t0
            ops = dict(qct.count_ops())
            two_q = sum(ops.get(k, 0) for k in ("ecr", "cz", "cx", "swap"))
            print(f"{n:>3}  {n+1:>6}  {n:>5}  "
                  f"{qct.depth():>6d}  {sum(ops.values()):>6d}  {two_q:>6d}  "
                  f"{elapsed:>8.3f}", flush=True)
        except BaseException as e:
            print(f"{n:>3}  {n+1:>6}  {n:>5}  FAILED  ({type(e).__name__})",
                  flush=True)


if __name__ == "__main__":
    main()
