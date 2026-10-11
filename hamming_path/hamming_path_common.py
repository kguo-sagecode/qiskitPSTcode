"""
Hamming-weighted path: the symmetric tridiagonal quotient matrix of the
distance partition of the binary hypercube Q_n.

The graph has n + 1 vertices labelled 0, 1, ..., n along a single path, with
edge weights
    w_k = sqrt(k * (n - k + 1))     for k = 1, ..., n
on the edge (k - 1, k). Eigenvalues of the weighted adjacency matrix are
n, n-2, ..., -n (the distinct eigenvalues of Q_n), and perfect state
transfer occurs from vertex 0 to vertex n at time t = pi/2.

Conventions used here:
    - One qubit per path vertex: vertex k <-> qubit k (Qiskit little-endian).
    - The vertex-encoding XY Hamiltonian, restricted to single excitations,
      equals the weighted adjacency matrix, as for the C_4 and Q_3
      scripts.
    - For PST from vertex 0 to vertex n, the initial single excitation is at
      qubit 0 and the target after t = pi/2 is the bit-string with bit n set
      (Qiskit little-endian -> the leftmost bit of an (n+1)-bit string).

PauliEvolution / weighted coefficients
--------------------------------------
`SparsePauliOp.from_list` accepts arbitrary complex coefficients on each
Pauli string. We pass the weights w_k * 0.5 directly; the
Suzuki--Trotter synthesis in `PauliEvolutionGate` then produces
`R_{XX}(w_k * t / r)` style rotations using the supplied coefficient as the
prefactor. There is no requirement that coefficients be 1 or +/- i.
"""

from __future__ import annotations

from math import sqrt
from typing import Iterable

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, UnitaryGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter
from scipy.linalg import expm


def hamming_weights(n: int) -> list[float]:
    """Edge weights w_k = sqrt(k * (n - k + 1)) for k = 1, ..., n."""
    return [sqrt(k * (n - k + 1)) for k in range(1, n + 1)]


def hamming_hamiltonian(n: int) -> SparsePauliOp:
    """
    Weighted XY Hamiltonian on the Hamming-weighted path of n+1 vertices:

        H  =  (1/2) * sum_{k=1}^n w_k * (X_{k-1} X_k + Y_{k-1} Y_k),

    written as a SparsePauliOp in Qiskit's little-endian Pauli-string
    convention (rightmost character = qubit 0).
    """
    N = n + 1
    weights = hamming_weights(n)
    terms = []
    for k, w in enumerate(weights, start=1):
        i, j = k - 1, k    # edge (k-1, k)
        for P in ("X", "Y"):
            lab = ["I"] * N
            lab[i] = P
            lab[j] = P
            terms.append(("".join(reversed(lab)), 0.5 * w))
    return SparsePauliOp.from_list(terms)


def option_a_circuit(
    n: int,
    t: float,
    not_qubit: int = 0,
    twice: bool = False,
) -> QuantumCircuit:
    """
    Option A: build U(t) = e^{-itH} as a dense 2^(n+1) x 2^(n+1) matrix on
    the full Hilbert space of the n+1 qubits (32 x 32 for n = 4) and append
    it as a UnitaryGate.

    If `twice` is True the circuit applies U(t) twice in sequence (so for
    t = pi/4 this gives U(pi/2)).
    """
    N = n + 1
    H = hamming_hamiltonian(n).to_matrix()
    U = expm(-1j * t * H)
    qc = QuantumCircuit(N)
    qc.x(not_qubit)
    qc.append(UnitaryGate(U, label=f"U_HP{n}(t)"), range(N))
    if twice:
        qc.append(UnitaryGate(U, label=f"U_HP{n}(t)"), range(N))
    return qc


def option_b_circuit(
    n: int,
    t: float,
    reps: int,
    not_qubit: int = 0,
    twice: bool = False,
    order: int = 2,
) -> QuantumCircuit:
    """
    Option B: Trotterised evolution via PauliEvolutionGate.

    The weighted SparsePauliOp is passed directly; PauliEvolutionGate
    accepts arbitrary real (or complex) coefficients, so the irrational
    weights w_k = sqrt(k*(n-k+1)) are baked into the per-rep rotation
    angles automatically.
    """
    N = n + 1
    H = hamming_hamiltonian(n)
    evo = PauliEvolutionGate(H, time=t, synthesis=SuzukiTrotter(order=order, reps=reps))
    qc = QuantumCircuit(N)
    qc.x(not_qubit)
    qc.append(evo, range(N))
    if twice:
        qc.append(evo, range(N))
    return qc


def target_bitstring(n: int, target_qubit: int | None = None) -> str:
    """
    Bit-string for the single-excitation state at `target_qubit` (default
    qubit n, the antipodal vertex on the Hamming-weighted path).
    """
    N = n + 1
    if target_qubit is None:
        target_qubit = n
    return format(1 << target_qubit, f"0{N}b")


def single_excitation_bitstrings(n: int) -> list[str]:
    """List of the n+1 single-excitation bit-strings, indexed by vertex."""
    N = n + 1
    return [format(1 << k, f"0{N}b") for k in range(N)]
