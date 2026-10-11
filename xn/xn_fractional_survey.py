"""
Transpilation and noiseless simulation of the X_n peak-state-transfer
circuits, in the default cz basis and with fractional gates (rzz) enabled,
against the ibm_kingston targets.

For each n = 2..11 at r*(n) (chosen for the cz model in xn_pst_r_optimal.py
and reused for the fractional circuits; report, Section 8.4):

  - transpile Option B (PauliEvolutionGate, t = pi) at opt level 3,
    seed 0, to the cz target and to the fractional target (+FoldRzzAngle);
  - record depth / cz / rzz / n_2q for both;
  - compute exact noiseless per-vertex probabilities for both bases and
    check they agree;
  - optionally, compare against per-vertex values in a LaTeX table (set
    APPENDIX_TEX to its path; the parser expects the layout of the
    report's appendix tables).

Requires an IBM Quantum account to fetch the backend target; nothing is
submitted. Results: printed table + xn_fractional_survey.json.
"""

from __future__ import annotations

import json
import re
from math import pi

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.transpiler.passes.basis import FoldRzzAngle

from xn_path_common import option_b_circuit, target_qubit

R_PER_N = {2: 5, 3: 6, 4: 7, 5: 8, 6: 4, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8}
T = pi
RESULTS_FILE = "xn_fractional_survey.json"
APPENDIX_TEX = None   # optional: path to a LaTeX file with the per-vertex tables


def fold_and_clean(qct: QuantumCircuit) -> QuantumCircuit:
    folded = PassManager([FoldRzzAngle()]).run(qct)
    out = qct.copy_empty_like()
    for inst in folded.data:
        if inst.operation.name == "global_phase":
            out.global_phase += inst.operation.params[0]
        else:
            out.append(inst)
    return out


def per_vertex(isa: QuantumCircuit, n_qubits: int) -> dict[int, float]:
    """Exact noiseless single-excitation probabilities, keyed by clbit index."""
    active = sorted({isa.find_bit(q).index for ins in isa.data for q in ins.qubits
                     if ins.operation.name != "barrier"})
    ph2new = {p: i for i, p in enumerate(active)}
    uni = QuantumCircuit(len(active))
    meas: dict[int, int] = {}
    for ins in isa.data:
        if ins.operation.name == "measure":
            meas[isa.find_bit(ins.clbits[0]).index] = \
                ph2new[isa.find_bit(ins.qubits[0]).index]
        elif ins.operation.name != "barrier":
            uni.append(ins.operation, [ph2new[isa.find_bit(q).index] for q in ins.qubits])
    probs = np.abs(Statevector.from_instruction(uni).data) ** 2
    out = {k: 0.0 for k in range(n_qubits)}
    for idx, pr in enumerate(probs):
        cl = 0
        for cbit in range(n_qubits):
            if (idx >> meas[cbit]) & 1:
                cl |= 1 << cbit
        if cl and (cl & (cl - 1)) == 0:
            out[cl.bit_length() - 1] += pr
    return out


def parse_appendix() -> dict[int, dict[int, float]]:
    if not APPENDIX_TEX:
        return {}
    tex = open(APPENDIX_TEX).read()
    xn_part = tex.split("Per-vertex breakdowns for the $X_n$")[1]
    blocks: dict[int, dict[int, float]] = {}
    cur_n = None
    for line in xn_part.splitlines():
        m = re.search(r"multirow\{\d+\}\{\*\}\{\$(\d+)\$ \(\$r=\d+\$\)\}", line)
        if m:
            cur_n = int(m.group(1))
            blocks[cur_n] = {}
        row = re.search(r"&\s*\$\\mathbf\{(\d+)\}\$\s*&\s*\\texttt", line) \
            or re.search(r"&\s*\$(\d+)\$\s*&\s*\\texttt", line)
        if cur_n is not None and row:
            k = int(row.group(1))
            vals = re.findall(r"\$\\?m?a?t?h?b?f?\{?(0\.\d{6}|1\.000000)\}?\$", line)
            if vals:
                blocks[cur_n][k] = float(vals[0])
    return blocks


def main() -> None:
    appendix = parse_appendix()
    print("parsed appendix X_n blocks:",
          {n: len(v) for n, v in sorted(appendix.items())})

    service = QiskitRuntimeService()
    king_cz = service.backend("ibm_kingston", use_fractional_gates=False)
    king_fr = service.backend("ibm_kingston", use_fractional_gates=True)
    pm_cz = generate_preset_pass_manager(
        optimization_level=3, backend=king_cz, seed_transpiler=0
    )
    pm_fr = generate_preset_pass_manager(
        optimization_level=3, backend=king_fr, seed_transpiler=0
    )
    print("(transpile + Aer only, no hardware submission)\n")
    hdr = (f"{'n':>3s} {'r*':>3s} {'qb':>3s} {'2q':>5s} "
           f"{'depth_cz':>9s} {'depth_fr':>9s} {'cz_fr':>6s} {'rzz_fr':>7s} "
           f"{'P_target':>10s} {'bases':>6s} {'vs appendix':>12s}")
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for n in range(2, 12):
        r = R_PER_N[n]
        nq = 2 * n
        qc = option_b_circuit(n, T, reps=r)
        qc.measure_all()

        isa_cz = pm_cz.run(qc)
        isa_fr = fold_and_clean(pm_fr.run(qc))
        angles = [float(i.operation.params[0]) for i in isa_fr.data
                  if i.operation.name == "rzz"]
        assert all(0 < a <= pi / 2 + 1e-12 for a in angles), f"n={n}: rzz out of range"

        ops_cz = dict(isa_cz.count_ops())
        ops_fr = dict(isa_fr.count_ops())
        n2q_cz = ops_cz.get("cz", 0) + ops_cz.get("rzz", 0)
        n2q_fr = ops_fr.get("cz", 0) + ops_fr.get("rzz", 0)

        pv_cz = per_vertex(isa_cz, nq)
        pv_fr = per_vertex(isa_fr, nq)
        bases_match = max(abs(pv_cz[k] - pv_fr[k]) for k in pv_cz) < 5e-7
        if n in appendix:
            app_dev = max(abs(pv_fr[k] - v) for k, v in appendix[n].items())
            app_str = f"{app_dev:.6f}"
        else:
            app_str = "n/a"

        tq = target_qubit(n)
        results[n] = {
            "r": r, "qubits": nq,
            "cz_basis": {"depth": isa_cz.depth(), "cz": ops_cz.get("cz", 0),
                         "n2q": n2q_cz},
            "fractional": {"depth": isa_fr.depth(), "cz": ops_fr.get("cz", 0),
                           "rzz": ops_fr.get("rzz", 0), "n2q": n2q_fr},
            "P_target": pv_fr[tq],
            "per_vertex": pv_fr,
        }
        print(f"{n:>3d} {r:>3d} {nq:>3d} {n2q_cz:>5d} "
              f"{isa_cz.depth():>9d} {isa_fr.depth():>9d} "
              f"{ops_fr.get('cz', 0):>6d} {ops_fr.get('rzz', 0):>7d} "
              f"{pv_fr[tq]:>10.6f} {'OK' if bases_match else 'DIFF':>6s} {app_str:>12s}")
        if n2q_cz != n2q_fr:
            print(f"    NOTE: n2q differs between bases ({n2q_cz} vs {n2q_fr})")

    with open(RESULTS_FILE, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
