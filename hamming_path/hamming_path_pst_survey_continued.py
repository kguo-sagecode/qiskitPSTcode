"""
Continue the survey of hamming_path_pst_survey.py for n in {10, 11, 12},
catching the Rust-side PanicException that Quantum Shannon Decomposition
raises at large n (NoConvergence in its linear-algebra step).
"""

from __future__ import annotations

import time
from math import pi

from qiskit import transpile

from hamming_path_common import option_a_circuit, option_b_circuit

BASIS = ["cz", "sx", "rz", "x"]


def survey(qc) -> dict:
    t0 = time.time()
    qct = transpile(qc, basis_gates=BASIS, optimization_level=3, seed_transpiler=0)
    elapsed = time.time() - t0
    ops = dict(qct.count_ops())
    two_q = sum(ops.get(k, 0) for k in ("ecr", "cz", "cx", "swap"))
    return {
        "depth": qct.depth(),
        "total": sum(ops.values()),
        "two_q": two_q,
        "transpile_time": elapsed,
    }


def main() -> None:
    print(f"{'n':>3}  {'qubits':>6}  "
          f"{'A depth':>10}  {'A total':>10}  {'A 2q':>9}  {'A time(s)':>10}  "
          f"{'B depth':>8}  {'B total':>8}  {'B 2q':>7}  {'B time(s)':>9}",
          flush=True)
    print("-" * 120, flush=True)

    for n in (10, 11, 12):
        qubits = n + 1

        # Option A -- catch BaseException to handle PanicException from PyO3
        a_str = ""
        try:
            a = survey(option_a_circuit(n, pi / 2))
            a_str = (f"{a['depth']:>10d}  {a['total']:>10d}  {a['two_q']:>9d}  "
                     f"{a['transpile_time']:>10.2f}")
        except BaseException as e:  # noqa: BLE001  (need to catch panics from Rust)
            etype = type(e).__name__
            a_str = f"{'QSD ' + etype:>43s}"

        # Option B
        try:
            b = survey(option_b_circuit(n, pi / 2, reps=8))
            b_str = (f"{b['depth']:>8d}  {b['total']:>8d}  {b['two_q']:>7d}  "
                     f"{b['transpile_time']:>9.2f}")
        except BaseException as e:  # noqa: BLE001
            b_str = f"FAILED  ({type(e).__name__})"

        print(f"{n:>3}  {qubits:>6}  {a_str}  {b_str}", flush=True)


if __name__ == "__main__":
    main()
