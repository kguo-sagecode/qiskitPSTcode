"""
Quick verification of the X_n peak-state-transfer family code in
xn_path_common.py.

For each n in {2, 3, 4, 5} we print:
  - the edge weights along the path,
  - the eight largest eigenvalues of the full Hamiltonian, and the
    eigenvalues of its single-excitation block (should be +/-1, ..., +/-n),
  - noiseless P(target) at t = pi (peak time), under both Option A
    (exact UnitaryGate, evaluated directly) and Option B
    (PauliEvolutionGate, r = 8, evaluated after transpiling to
    {cz, sx, rz, x}, so that the Trotter error is included),
    and the theoretical peak amplitude
        B(A_n)_{w_1, w_n} = 2 * 4^(n-1) / (sqrt(n) * (n+1) * C_n)
    where C_n is the n-th Catalan number.
"""

from __future__ import annotations

from math import comb, pi, sqrt

import numpy as np

from qiskit import transpile
from qiskit.quantum_info import Statevector

from xn_path_common import (
    option_a_circuit,
    option_b_circuit,
    single_excitation_bitstrings,
    target_qubit,
    xn_hamiltonian,
    xn_weights,
)


BASIS = ["cz", "sx", "rz", "x"]


def catalan(n: int) -> int:
    return comb(2 * n, n) // (n + 1)


def peak_amplitude_theory(n: int) -> float:
    """B(A_n)_{w_1, w_n} from Theorem 6.5 of Coutinho, Guo and Schmeits (arXiv:2505.11986)."""
    return 2 * 4 ** (n - 1) / (sqrt(n) * (n + 1) * catalan(n))


def report(n: int) -> None:
    print("\n" + "=" * 74)
    print(f"  X_{n}    ({2 * n} qubits,  source = qubit 1,  target = qubit {2*n-1})")
    print("=" * 74)

    weights = xn_weights(n)
    print(f"  edge weights along the path ({2*n - 1} edges):")
    print("    " + ", ".join(f"{w:.6f}" for w in weights))

    # Single-excitation spectrum from the SparsePauliOp matrix
    H = xn_hamiltonian(n).to_matrix()
    eigvals = sorted(np.linalg.eigvalsh(H), reverse=True)
    print(f"  full spectrum (top 8): "
          + ", ".join(f"{x:.4f}" for x in eigvals[:8]))

    # Restrict to single-excitation block manually
    se_indices = [1 << k for k in range(2 * n)]
    H_se = H[np.ix_(se_indices, se_indices)].real
    se_eigvals = sorted(np.linalg.eigvalsh(H_se), reverse=True)
    print(f"  single-excitation spectrum: "
          + ", ".join(f"{x:.4f}" for x in se_eigvals))

    # Theoretical peak amplitude
    B = peak_amplitude_theory(n)
    print(f"  theoretical peak amplitude B(A_n)_{{w_1, w_n}}: {B:.6f}")
    print(f"    peak probability:                            {B*B:.6f}")
    print(f"    pi/4 (asymptotic):                           {pi/4:.6f}")

    # Noiseless statevectors. Statevector.from_instruction evaluates an
    # untranspiled PauliEvolutionGate exactly (no Trotter error), so the
    # Option B circuit is transpiled first.
    qc_a = option_a_circuit(n, pi)
    sv_a = Statevector.from_instruction(qc_a)
    p_a = np.abs(np.asarray(sv_a.data)) ** 2
    qc_b = transpile(option_b_circuit(n, pi, reps=8), basis_gates=BASIS,
                     optimization_level=3, seed_transpiler=0)
    sv_b = Statevector.from_instruction(qc_b)
    p_b = np.abs(np.asarray(sv_b.data)) ** 2

    tq = target_qubit(n)
    tgt_idx = 1 << tq
    se_set = {1 << k for k in range(2 * n)}
    in_sub_a = float(sum(p_a[i] for i in se_set))
    in_sub_b = float(sum(p_b[i] for i in se_set))
    print(f"  noiseless P(target = qubit {tq}):")
    print(f"    Option A (exact UnitaryGate):      {p_a[tgt_idx]:.6f}    "
          f"(in-subspace = {in_sub_a:.6f})")
    print(f"    Option B (Trotter r=8):            {p_b[tgt_idx]:.6f}    "
          f"(in-subspace = {in_sub_b:.6f})")


def main() -> None:
    for n in (2, 3, 4, 5):
        report(n)


if __name__ == "__main__":
    main()
