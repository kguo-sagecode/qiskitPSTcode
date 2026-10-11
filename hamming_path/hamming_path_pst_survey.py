"""
Gate-count survey for the Hamming-weighted paths at t = pi/2, both
Option A (UnitaryGate / Quantum Shannon Decomposition) and Option B
(PauliEvolutionGate + Suzuki-Trotter, r=8), for n = 2, 3, ..., 12.

Transpiled to basis {cz, sx, rz, x} with optimization_level=3, no
coupling map. Reports depth, total ops, two-qubit (cz) count, and transpile
wall time. Quantum Shannon Decomposition fails at n = 10 (a Rust-side
PanicException, which is caught; see also
hamming_path_pst_survey_continued.py).

Option A's two-qubit gate count grows as ~(23/48) * 4^(n+1); somewhere
around n = 8 or 9 the QSD synthesis is no longer practical (gate counts in
the tens of thousands, transpile minutes). The script attempts Option A
for each n, and skips it for all remaining n once one attempt fails or
takes longer than `OPT_A_TIME_LIMIT` seconds.
"""

from __future__ import annotations

import time
from math import pi

from qiskit import transpile

from hamming_path_common import option_a_circuit, option_b_circuit


BASIS = ["cz", "sx", "rz", "x"]
OPT_A_TIME_LIMIT = 600.0   # seconds; if previous Option A took longer, skip the next


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


def fmt_a(row: dict | None) -> str:
    if row is None:
        return f"{'skipped':>8s}  {'':>8s}  {'':>7s}  {'':>9s}"
    return (f"{row['depth']:>8d}  {row['total']:>8d}  {row['two_q']:>7d}  "
            f"{row['transpile_time']:>9.2f}")


def fmt_b(row: dict) -> str:
    return (f"{row['depth']:>8d}  {row['total']:>8d}  {row['two_q']:>7d}  "
            f"{row['transpile_time']:>9.2f}")


def main() -> None:
    print(f"{'n':>3}  {'qubits':>6}  "
          f"{'A depth':>8}  {'A total':>8}  {'A 2q':>7}  {'A time(s)':>9}  "
          f"{'B depth':>8}  {'B total':>8}  {'B 2q':>7}  {'B time(s)':>9}",
          flush=True)
    print("-" * 110, flush=True)

    last_opt_a_time = 0.0
    for n in range(2, 13):
        qubits = n + 1

        # Option A: skip if the previous run was over the time limit
        if last_opt_a_time > OPT_A_TIME_LIMIT:
            a_row = None
            print(f"{n:>3}  {qubits:>6}  " + fmt_a(a_row) + "  ", end="", flush=True)
        else:
            try:
                a_row = survey(option_a_circuit(n, pi / 2))
                last_opt_a_time = a_row["transpile_time"]
            except BaseException as e:  # noqa: BLE001  (QSD panics are not Exceptions)
                print(f"  (Option A failed for n={n}: {type(e).__name__})", flush=True)
                a_row = None
                last_opt_a_time = float("inf")
            print(f"{n:>3}  {qubits:>6}  " + fmt_a(a_row) + "  ", end="", flush=True)

        # Option B: always do
        try:
            b_row = survey(option_b_circuit(n, pi / 2, reps=8))
            print(fmt_b(b_row), flush=True)
        except BaseException as e:  # noqa: BLE001
            print(f"FAILED  ({type(e).__name__})", flush=True)


if __name__ == "__main__":
    main()
