"""
Gate-count survey for the X_n peak-state-transfer circuits at t = pi.

For each n in 2..20:
  - Option A (UnitaryGate + Quantum Shannon Decomposition) is attempted
    until QSD either becomes impractical (gate count blows up) or numerically
    fails (Rust-side PanicException). Wrapped in BaseException to catch panics.
  - Option B (PauliEvolutionGate + second-order Suzuki--Trotter, r=8).

Transpile to basis {cz, sx, rz, x} with optimization_level=3, no coupling map.
Reports depth, total operations, two-qubit (cz) gate count, and transpile time.
"""

from __future__ import annotations

import time
from math import pi

from qiskit import transpile

from xn_path_common import option_a_circuit, option_b_circuit


BASIS = ["cz", "sx", "rz", "x"]
TROTTER_REPS = 8


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
        "time": elapsed,
    }


def fmt(row: dict | None) -> str:
    if row is None:
        return f"{'skipped':>10s}  {'':>10s}  {'':>8s}  {'':>9s}"
    return (f"{row['depth']:>10d}  {row['total']:>10d}  {row['two_q']:>8d}  "
            f"{row['time']:>9.2f}")


def main() -> None:
    print(
        f"{'n':>3}  {'qubits':>6}  "
        f"{'A depth':>10}  {'A total':>10}  {'A 2q':>8}  {'A time(s)':>9}  "
        f"{'B depth':>10}  {'B total':>10}  {'B 2q':>8}  {'B time(s)':>9}",
        flush=True,
    )
    print("-" * 130, flush=True)

    OPT_A_TIME_LIMIT = 60.0
    last_a_time = 0.0

    for n in range(2, 21):
        qubits = 2 * n

        # Option A: try until it's too slow or panics
        if last_a_time > OPT_A_TIME_LIMIT:
            a_row = None
        else:
            try:
                a_row = survey(option_a_circuit(n, pi))
                last_a_time = a_row["time"]
            except BaseException as e:
                a_row = None
                print(f"  (Option A at n={n} failed: {type(e).__name__})", flush=True)
                last_a_time = float("inf")

        # Option B: always do
        try:
            b_row = survey(option_b_circuit(n, pi, reps=TROTTER_REPS))
        except BaseException as e:
            print(f"  (Option B at n={n} failed: {type(e).__name__})", flush=True)
            continue

        print(f"{n:>3}  {qubits:>6}  {fmt(a_row)}  {fmt(b_row)}", flush=True)


if __name__ == "__main__":
    main()
