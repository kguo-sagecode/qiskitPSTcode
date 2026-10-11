"""
SWAP-chain baseline on ibm_kingston, n = 2..21 (report, Section 9).

Builds X(q_0); SWAP(0,1); SWAP(1,2); ...; SWAP(n-2,n-1); measure_all,
transpiles to ibm_kingston at optimization_level=0 (at level 3 the
transpiler replaces the SWAPs by a relabelling of the qubits), and submits
at 4096 shots per job.

The n qubits are placed on the first n qubits of a fixed connected line of
21 physical qubits (via initial_layout), so each SWAP costs exactly 3 CZs
and the transpiler does not insert routing SWAPs.

For each n we report depth, total_ops, 2q (CZ) count, in-subspace fraction,
P(target) (the bit-string with bit n-1 set), and job.usage() QPU seconds.

By default the script only transpiles (dry run). With --submit it submits
20 jobs to IBM Quantum hardware and uses QPU time.
"""

from __future__ import annotations

import argparse
import sys

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


N_RANGE = range(2, 22)
SHOTS = 4096
BACKEND_NAME = "ibm_kingston"


def find_heavy_hex_line(backend, length: int) -> list[int]:
    """Greedy DFS for a length-`length` simple path in the backend's coupling map."""
    cmap = backend.coupling_map
    # Build undirected adjacency.
    adj = {q: set() for q in range(backend.num_qubits)}
    for a, b in cmap.get_edges():
        adj[a].add(b)
        adj[b].add(a)

    # DFS from each start qubit. Heavy-hex always contains long simple paths;
    # we just need any one of length `length`.
    def dfs(path: list[int]) -> list[int] | None:
        if len(path) == length:
            return path
        last = path[-1]
        for nb in adj[last]:
            if nb in path:
                continue
            found = dfs(path + [nb])
            if found is not None:
                return found
        return None

    for start in range(backend.num_qubits):
        result = dfs([start])
        if result is not None:
            return result
    raise RuntimeError(f"No connected line of length {length} on {backend.name}")


def build_chain(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"swap_chain_n{n}")
    qc.x(0)
    for k in range(n - 1):
        qc.swap(k, k + 1)
    qc.measure_all()
    return qc


def target_bitstring(n: int) -> str:
    return "1" + "0" * (n - 1)


def single_excitation_bitstrings(n: int) -> list[str]:
    return [format(1 << k, f"0{n}b") for k in range(n)]


def run_one(n: int, line: list[int], backend, *, dry_run: bool) -> None:
    print("\n" + "=" * 78)
    print(f"  SWAP chain n={n}  ({n} qubits, {n-1} SWAPs, target = '{target_bitstring(n)}')")
    print("=" * 78)

    qc = build_chain(n)
    layout = line[:n]
    pm = generate_preset_pass_manager(
        optimization_level=0,
        backend=backend,
        initial_layout=layout,
        seed_transpiler=0,
    )
    isa = pm.run(qc)
    ops = dict(isa.count_ops())
    two_q = sum(ops.get(k, 0) for k in ("cz", "ecr", "cx", "swap"))
    print(f"  layout (physical qubits used): {layout}")
    print(f"  ISA: depth={isa.depth()} total_ops={sum(ops.values())} 2q={two_q}  ops={ops}")

    if dry_run:
        print("  [DRY RUN -- not submitting]")
        return

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = SHOTS
    sampler.options.dynamical_decoupling.enable = False
    job = sampler.run([isa])
    print(f"  job ID: {job.job_id()}", flush=True)

    result = job.result()
    counts = result[0].data.meas.get_counts()
    total = sum(counts.values())

    tgt_bit = target_bitstring(n)
    se = set(single_excitation_bitstrings(n))
    in_sub = sum(c for b, c in counts.items() if b in se) / total
    p_target = counts.get(tgt_bit, 0) / total
    usage = job.usage()

    print(f"\n  shots: {total}")
    print(f"  in-subspace fraction: {in_sub:.6f}")
    print(f"  P({tgt_bit}, target = q_{n-1}): {p_target:.6f}")
    print(f"  QPU usage: {usage} s")

    print(f"\n  Single-excitation rows ({n}):")
    for k in range(n):
        bit = format(1 << k, f"0{n}b")
        c = counts.get(bit, 0)
        marker = "  *TARGET*" if k == n - 1 else ""
        if k == 0:
            marker += "  (source)"
        print(f"    qubit {k:>2}    {bit}    {c}  ({c/total:.6f}){marker}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true",
                        help="Actually submit to hardware (default: dry-run)")
    args = parser.parse_args()
    dry_run = not args.submit

    service = QiskitRuntimeService()
    backend = service.backend(BACKEND_NAME)
    print(f"Backend: {backend.name}  ({backend.num_qubits} qubits)")
    if dry_run:
        print("MODE: dry-run (transpile only, no submission)\n")
    else:
        print("MODE: SUBMIT -- will spend QPU time\n")

    # Find one connected heavy-hex line of length 21 and reuse its prefix
    # for all smaller n, so the experiments live on the same physical qubits.
    line = find_heavy_hex_line(backend, max(N_RANGE))
    print(f"21-qubit heavy-hex line: {line}\n")

    for n in N_RANGE:
        try:
            run_one(n, line, backend, dry_run=dry_run)
        except Exception as e:
            print(f"  !! n={n} failed: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
