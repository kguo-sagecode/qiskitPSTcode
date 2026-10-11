"""
Demonstration of the Hamming-weighted path implementations:
  - the (n+1)-vertex quotient of the n-cube Q_n,
  - both Option A (UnitaryGate) and Option B (PauliEvolutionGate +
    Suzuki-Trotter), the latter using weighted SparsePauliOp coefficients,
  - at the three time settings t = pi/4 (binomial mixing on the path),
    t = pi/2 (PST), and (pi/4) twice (PST via the binomial-mixing intermediate).

This script does not submit anything to hardware. It just builds the
circuits, transpiles them to a basis (without coupling map), prints the
gate counts, and reports the noiseless single-excitation probabilities on
the (n+1) path vertices, computed from the transpiled circuits (so the
Option B values include the Trotter error).
"""

from __future__ import annotations

from math import pi

import numpy as np

from qiskit import transpile
from qiskit.quantum_info import Statevector

from hamming_path_common import (
    hamming_hamiltonian,
    hamming_weights,
    option_a_circuit,
    option_b_circuit,
    single_excitation_bitstrings,
)


BASIS = ["cz", "sx", "rz", "x"]


def gate_summary(qct) -> str:
    ops = dict(qct.count_ops())
    two_q = sum(ops.get(k, 0) for k in ("ecr", "cz", "cx", "swap"))
    return (
        f"depth={qct.depth():>4d}  total={sum(ops.values()):>5d}  2q={two_q:>4d}  "
        f"ops={ops}"
    )


def single_exc_probs(qct) -> list[float]:
    sv = Statevector.from_instruction(qct)
    probs = np.abs(np.asarray(sv.data)) ** 2
    se_bits = single_excitation_bitstrings(qct.num_qubits - 1)
    out = []
    for bit in se_bits:
        out.append(float(probs[int(bit, 2)]))
    return out


def report_n(n: int) -> None:
    print("\n" + "=" * 78)
    print(f"  Hamming-weighted path quotient of Q_{n}    (n+1 = {n+1} qubits)")
    print("=" * 78)
    weights = hamming_weights(n)
    H = hamming_hamiltonian(n)
    eigvals_sorted = sorted(np.linalg.eigvalsh(H.to_matrix()), reverse=True)
    print(f"  edge weights w_k = sqrt(k(n-k+1)) for k=1..{n}:")
    print("    " + ", ".join(f"{w:.6f}" for w in weights))
    print(f"  H is a {2**(n+1)}x{2**(n+1)} operator with "
          f"{len(H)} Pauli terms")
    print(f"  spectrum (full Hilbert space, top 8): "
          + ", ".join(f"{x:.4f}" for x in eigvals_sorted[:8]))

    # Trotter reps chosen to mirror the C_4 conventions
    reps_pi4 = 4
    reps_pi2 = 8

    configs = [
        ("Option A, t=pi/4",                 option_a_circuit(n, pi / 4)),
        ("Option B, t=pi/4, r=4",            option_b_circuit(n, pi / 4, reps=reps_pi4)),
        ("Option A, t=pi/2",                 option_a_circuit(n, pi / 2)),
        ("Option B, t=pi/2, r=8",            option_b_circuit(n, pi / 2, reps=reps_pi2)),
        ("Option A, (pi/4) twice",           option_a_circuit(n, pi / 4, twice=True)),
        ("Option B, (pi/4) twice, r=4 each", option_b_circuit(n, pi / 4, reps=reps_pi4, twice=True)),
    ]

    transpiled = [
        (label, transpile(qc, basis_gates=BASIS, optimization_level=3, seed_transpiler=0))
        for label, qc in configs
    ]

    print(f"\n  Transpile (basis only, no coupling map):")
    for label, qct in transpiled:
        print(f"    {label:<36s}  {gate_summary(qct)}")

    print(f"\n  Noiseless single-excitation probabilities on the {n+1} path vertices:")
    print(f"    vertex k  ->  probability  (target for PST is vertex {n})")
    for label, qct in transpiled:
        probs = single_exc_probs(qct)
        in_sub = sum(probs)
        peaks = ", ".join(
            f"k={k}:{p:.4f}" for k, p in enumerate(probs)
        )
        print(f"    {label}")
        print(f"      [{peaks}]   in-sub={in_sub:.6f}")


def main() -> None:
    for n in (2, 3, 4):
        report_n(n)


if __name__ == "__main__":
    main()
