"""
SWAP-chain baseline: move a single excitation from qubit 0 to qubit n-1
with a sequence of nearest-neighbour SWAPs.

For each n in 2..21 we build the circuit
    X(q_0); SWAP(q_0,q_1); SWAP(q_1,q_2); ...; SWAP(q_{n-2},q_{n-1})
and measure. The noiseless outcome is the bit-string with bit n-1 set
(in Qiskit little-endian: a leading '1' followed by n-1 zeros).

This file:
  - builds the circuit (build_chain)
  - prints text drawings for small n
  - runs Aer at 4096 shots for n = 2..21 and reports in-subspace fraction
    and P(target).
No hardware submission.
"""

from __future__ import annotations

import time
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_RANGE = range(2, 22)              # n = 2..21
SHOTS = 4096


def build_chain(n: int) -> QuantumCircuit:
    """X(q_0) then SWAP(k, k+1) for k = 0..n-2, then measure_all."""
    qc = QuantumCircuit(n, name=f"swap_chain_n{n}")
    qc.x(0)
    for k in range(n - 1):
        qc.swap(k, k + 1)
    qc.measure_all()
    return qc


def target_bitstring(n: int) -> str:
    """Qiskit little-endian: bit n-1 set -> '1' followed by (n-1) '0's."""
    return "1" + "0" * (n - 1)


def single_excitation_bitstrings(n: int) -> list[str]:
    """All n single-excitation bit-strings, indexed by qubit position."""
    return [format(1 << k, f"0{n}b") for k in range(n)]


def print_drawing(n: int) -> None:
    """Print an ASCII drawing of the chain for n qubits."""
    qc = build_chain(n)
    print(f"\n--- n = {n} ({n} qubits, {n-1} SWAPs, target = q_{n-1} = '{target_bitstring(n)}') ---")
    print(qc.draw(output="text", fold=110))


def run_aer(n: int) -> dict:
    qc = build_chain(n)
    sim = AerSimulator()
    qct = transpile(qc, sim, seed_transpiler=0)
    t0 = time.time()
    counts = sim.run(qct, shots=SHOTS, seed_simulator=0).result().get_counts()
    t_aer = time.time() - t0
    se = set(single_excitation_bitstrings(n))
    in_sub = sum(c for b, c in counts.items() if b in se) / SHOTS
    p_target = counts.get(target_bitstring(n), 0) / SHOTS
    return {"n": n, "in_sub": in_sub, "p_target": p_target, "t_aer": t_aer}


def main() -> None:
    print("=" * 78)
    print("SWAP-CHAIN BASELINE - circuit drawings for n = 2, 3, 5")
    print("=" * 78)
    for n in (2, 3, 5):
        print_drawing(n)

    print("\n" + "=" * 78)
    print(f"AER SWEEP  n = {N_RANGE.start}..{N_RANGE.stop-1},  shots = {SHOTS}")
    print("=" * 78)
    print(f"\n{'n':>3} {'target':>22} {'in-sub':>8} {'P(target)':>10} {'aer (s)':>8}")
    print("-" * 58)
    total_wall = 0.0
    for n in N_RANGE:
        r = run_aer(n)
        total_wall += r["t_aer"]
        print(f"{r['n']:>3} {target_bitstring(r['n']):>22} "
              f"{r['in_sub']:>8.4f} {r['p_target']:>10.6f} {r['t_aer']:>7.3f}s")
    print(f"\ntotal Aer wall-clock: {total_wall:.2f}s")


if __name__ == "__main__":
    main()
