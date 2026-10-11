"""
Retrieve a finished IBM Quantum job and emit a complete probability
distribution over all 2^N bitstrings, formatted as ready-to-paste LaTeX
table rows (six decimal places).

USAGE:
    python retrieve_job_full.py <JOB_ID> [-n N]

If -n is omitted the script infers N from the bitstring length in the
returned counts. The output is two blocks:
    (a) a sorted list of bitstring -> probability, with single-excitation
        markers;
    (b) a LaTeX \\texttt{...} & $0.xxxxxx$ row block (one per bitstring).
"""

from __future__ import annotations

import argparse
import json

from qiskit_ibm_runtime import QiskitRuntimeService


def single_exc_bitstrings(n: int) -> set[str]:
    return {format(1 << k, f"0{n}b") for k in range(n)}


def fetch_counts(job_id: str) -> dict[str, int]:
    service = QiskitRuntimeService()
    job = service.job(job_id)
    if str(job.status()) not in ("DONE", "JobStatus.DONE"):
        raise SystemExit(f"Job {job_id} is not finished (status: {job.status()}).")
    result = job.result()
    return result[0].data.meas.get_counts()


def distribution(counts: dict[str, int], n: int) -> dict[str, float]:
    total = sum(counts.values())
    dist = {format(idx, f"0{n}b"): 0.0 for idx in range(2 ** n)}
    for bit, c in counts.items():
        if len(bit) != n:
            raise ValueError(
                f"Bitstring {bit!r} has length {len(bit)}, expected {n}. "
                "Pass the correct -n, or check that the job is on the expected backend."
            )
        dist[bit] = c / total
    return dist


def print_human(dist: dict[str, float], n: int) -> None:
    se = single_exc_bitstrings(n)
    for bit in sorted(dist.keys()):
        marker = "  single-exc" if bit in se else ""
        print(f"  {bit}   {dist[bit]:.6f}{marker}")
    in_sub = sum(dist[b] for b in se)
    print(f"  -- in-subspace sum: {in_sub:.6f}")


def print_latex(dist: dict[str, float]) -> None:
    print()
    print("LaTeX rows (paste into the relevant column of the hardware table):")
    print()
    for bit in sorted(dist.keys()):
        print(f"\\texttt{{{bit}}} & ... & ${dist[bit]:.6f}$ \\\\")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("-n", type=int, default=None, help="number of qubits (inferred if omitted)")
    ap.add_argument("--save-json", action="store_true",
                    help="also save full distribution to job_<ID>_dist.json")
    args = ap.parse_args()

    counts = fetch_counts(args.job_id)
    sample_bit = next(iter(counts))
    n = args.n or len(sample_bit)
    print(f"Job: {args.job_id}    inferred N = {n}    shots = {sum(counts.values())}")
    dist = distribution(counts, n)
    print_human(dist, n)
    print_latex(dist)

    if args.save_json:
        with open(f"job_{args.job_id}_dist.json", "w") as f:
            json.dump(dist, f, indent=2, sort_keys=True)
        print(f"\nSaved {f'job_{args.job_id}_dist.json'}")


if __name__ == "__main__":
    main()
