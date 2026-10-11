"""
The X_n family of weighted paths with peak state transfer, from
G. Coutinho, K. Guo and V. Schmeits, "Peak state transfer in continuous
quantum walks", arXiv:2505.11986 (2025), Section 6.3.

The weighted graph X_n is the path on 2n vertices that visits, in order,
    v_n, w_1, v_{n-1}, w_2, ..., v_2, w_{n-1}, v_1, w_n,
with the edge between v_j and w_k carrying weight sqrt(j*k). Equivalently,
relabelling along the path 0, 1, 2, ..., 2n-1 in visit order:
  - path position 2j     (j = 0, ..., n-1) is v_{n-j},
  - path position 2j + 1 (j = 0, ..., n-1) is w_{j+1},
  - the edge between path positions k and k+1 has weight beta_{k+1},
    where beta_e (1-indexed) is
        if e = 2j-1 (odd):  sqrt(j * (n - j + 1))    [edge v_{n-j+1} -- w_j]
        if e = 2j   (even): sqrt(j * (n - j))        [edge w_j -- v_{n-j}]

The adjacency matrix A_n has eigenvalues +/-1, +/-2, ..., +/-n, each
simple. The walk has peak state transfer from w_1 (qubit 1) to w_n
(qubit 2n-1) at time t = pi; the peak transfer amplitude B(A_n)_{w_1, w_n}
decreases to sqrt(pi)/2 (so the peak probability decreases to
pi/4 ~ 0.785) as n grows.

Conventions (as in ../hamming_path/hamming_path_common.py):
  - 2n qubits, indexed 0..2n-1, with qubit k = path position k.
  - Source: qubit 1 (w_1). Target: qubit 2n-1 (w_n).
  - The walk runs in the single-excitation subspace of the weighted XY
    Hamiltonian.
"""

from __future__ import annotations

from math import sqrt
from typing import List

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, UnitaryGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from scipy.linalg import expm


def xn_weights(n: int) -> List[float]:
    """
    Edge weights beta_1, ..., beta_{2n-1} along the X_n path (1-indexed), in
    visit order from path position 0 to 2n-1.

    Edge e between path positions e-1 and e:
      e = 2j - 1 (odd):  sqrt(j * (n - j + 1))
      e = 2j     (even): sqrt(j * (n - j))
    """
    weights: List[float] = []
    for e in range(1, 2 * n):
        if e % 2 == 1:
            j = (e + 1) // 2
            weights.append(sqrt(j * (n - j + 1)))
        else:
            j = e // 2
            weights.append(sqrt(j * (n - j)))
    return weights


def xn_hamiltonian(n: int) -> SparsePauliOp:
    """
    Weighted XY Hamiltonian
        H = (1/2) sum_{k=0}^{2n-2} beta_{k+1} (X_k X_{k+1} + Y_k Y_{k+1})
    on 2n qubits, returned as a SparsePauliOp in Qiskit little-endian.
    """
    N = 2 * n
    weights = xn_weights(n)
    terms = []
    for k, w in enumerate(weights):
        i, j = k, k + 1
        for P in ("X", "Y"):
            lab = ["I"] * N
            lab[i] = P
            lab[j] = P
            terms.append(("".join(reversed(lab)), 0.5 * w))
    return SparsePauliOp.from_list(terms)


def source_qubit(n: int) -> int:
    """Path position of the source vertex w_1."""
    return 1


def target_qubit(n: int) -> int:
    """Path position of the target vertex w_n."""
    return 2 * n - 1


def target_bitstring(n: int) -> str:
    """Qiskit little-endian bit-string for the target single-excitation state."""
    N = 2 * n
    return format(1 << target_qubit(n), f"0{N}b")


def single_excitation_bitstrings(n: int) -> List[str]:
    """List of all 2n single-excitation bit-strings, indexed by qubit."""
    N = 2 * n
    return [format(1 << k, f"0{N}b") for k in range(N)]


def option_a_circuit(n: int, t: float) -> QuantumCircuit:
    """
    Option A: build U(t) = e^{-itH} as a dense (2^(2n))-square matrix and
    append it as a UnitaryGate.  Feasible up to about n = 5 (10 qubits,
    1024 x 1024 matrix) before Quantum Shannon Decomposition becomes
    impractical (cf. ../hamming_path/hamming_path_pst_survey.py).
    """
    N = 2 * n
    H = xn_hamiltonian(n).to_matrix()
    U = expm(-1j * t * H)
    qc = QuantumCircuit(N)
    qc.x(source_qubit(n))
    qc.append(UnitaryGate(U, label=f"U_X{n}"), range(N))
    return qc


def option_b_circuit(
    n: int,
    t: float,
    reps: int = 8,
    order: int = 2,
) -> QuantumCircuit:
    """
    Option B: Trotterised evolution via PauliEvolutionGate.  The weighted
    SparsePauliOp is passed directly; PauliEvolutionGate handles arbitrary
    real coefficients, so the weights beta_e = sqrt(j*k) enter the
    per-edge rotation angles directly.
    """
    N = 2 * n
    H = xn_hamiltonian(n)
    evo = PauliEvolutionGate(H, time=t, synthesis=SuzukiTrotter(order=order, reps=reps))
    qc = QuantumCircuit(N)
    qc.x(source_qubit(n))
    qc.append(evo, range(N))
    return qc
