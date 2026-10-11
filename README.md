# qiskitPSTcode

This is the accompanying code to our paper *Perfect and peak state
transfer on IBM quantum hardware* by Priscilla Awuah and Krystal Guo. We
will add the arXiv number in the next release.

The scripts run continuous-time quantum walks with perfect state transfer
(PST) and peak state transfer on IBM Quantum hardware and on the Aer
simulator, and produce the tables and figures of the paper.

All hardware runs in the paper were made on `ibm_kingston`, a 156-qubit
IBM Heron r2 processor, with 4096 shots per job and no error mitigation.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code was tested with Python 3.11 and the versions in `requirements.txt`.

Many scripts run locally. The others need an IBM Quantum account: the
`hardware_*.py` scripts submit jobs, and a few scripts fetch the
`ibm_kingston` backend to transpile against it without submitting
anything. To store your credentials once:

```
export IBM_QUANTUM_TOKEN="<your API key>"
export IBM_QUANTUM_INSTANCE="<your instance CRN>"
python save_ibm_account.py
```

**Running a `hardware_*.py` script submits jobs to IBM Quantum hardware
and uses QPU time.** The docstring of each script states how many jobs it
submits. `swap_chain/hardware_swap_chain_baseline.py` only transpiles
unless it is given `--submit`.

## Conventions

- There is one qubit for each vertex of the graph. The walk starts from a
  single excitation on the source qubit (an $X$ gate applied to
  $\lvert 0\cdots 0\rangle$), evolves under the XY Hamiltonian
  $H = \tfrac12 \sum_{uv \in E} J_{uv}(X_u X_v + Y_u Y_v)$, and ends with a
  measurement of all qubits. On the single-excitation subspace, $H$ acts as
  the weighted adjacency matrix of the graph.
- Bit-strings are in Qiskit's little-endian order: the rightmost bit is
  qubit 0, so the state with the excitation on qubit $k$ is the bit-string
  with bit $k$ set.
- *Option A* builds $e^{-itH}$ exactly as a matrix and appends it as a
  `UnitaryGate`. *Option B* uses `PauliEvolutionGate` with the second-order
  Suzuki–Trotter formula and $r$ repetitions (the Trotter rep count).
- Circuits are transpiled with `optimization_level=3` and
  `seed_transpiler=0`, either to the basis {cz, sx, rz, x} without a
  coupling map, or to the `ibm_kingston` target, in the default cz basis or
  with fractional gates (rzz, rx) enabled. In the fractional basis, the
  `FoldRzzAngle` pass brings all rzz angles into $[0, \pi/2]$.
- $n_{2q}$ is the number of two-qubit gates after transpilation. The
  survival fraction is $S = (1 - p_{2q})^{n_{2q}}$ with
  $p_{2q} = 2.0 \times 10^{-3}$, the median cz error of `ibm_kingston`. The
  predicted hardware target probability is $P_{\mathrm{noiseless}} \cdot S$,
  where $P_{\mathrm{noiseless}}$ is the exact noiseless target probability
  of the transpiled circuit. For each graph, $r^*$ is the value of $r$ that
  maximizes the predicted hardware target probability.

## Layout

Each folder corresponds to one section of the paper. Results are printed;
the scripts that also save them write JSON files to the current directory,
and figures go to a `figures/` folder next to the script. Run a script from
its folder, for example

```
cd hamming_path
python hamming_path_pst_r_optimal.py
```

In the tables below, *local* means the script runs on your machine,
*account* means it needs an IBM Quantum account but submits nothing, and
*hardware* means it submits jobs.

### `c4/`: the 4-cycle $C_4 = Q_2$ (Section 5)

PST between antipodal vertices at $t = \pi/2$, and uniform mixing at
$t = \pi/4$.

| Script | What it does | Needs |
|---|---|---|
| `hardware_option_a_pi_over_2.py`, `hardware_option_b_pi_over_2.py` | PST at $t = \pi/2$ with Option A and with Option B ($r = 8$), cz basis | hardware (1 job each) |
| `hardware_option_a.py`, `hardware_option_b.py` | uniform mixing at $t = \pi/4$ (Option B with $r = 4$), cz basis | hardware (1 job each) |
| `hardware_option_a_twice.py`, `hardware_option_b_twice.py` | PST as two blocks of $t = \pi/4$ (Option B with $r = 4$ per block), cz basis | hardware (1 job each) |
| `hardware_c4_fractional.py` | the same six circuits in the fractional basis | hardware (6 jobs) |
| `c4_aer_device_distributions.py` | exact noiseless distributions of the device-transpiled circuits | account |
| `c4_aer_fractional_distributions.py` | the same for the fractional basis, compared with the cz basis | account |
| `c4_fractional_transpile_survey.py` | gate counts of the six circuits in both bases | account |
| `c4_fractional_figure_and_table.py` | gate-count table for $t = \pi/2$, drawings of the fractional-basis circuits, median calibration errors | account |
| `c4_optA_transpiled_figure_fold80.py` | figure of the two transpiled $t = \pi/2$ circuits (cz basis) | account |

### `q3/`: the 3-cube $Q_3$ (Section 6)

| Script | What it does | Needs |
|---|---|---|
| `simulate_option_a_q3.py`, `simulate_option_b_q3.py` | $t = \pi/4$: gate counts without a coupling map, and an Aer run | local |
| `simulate_option_a_q3_pi_over_2.py`, `simulate_option_b_q3_pi_over_2.py` | the same at $t = \pi/2$ | local |
| `simulate_option_a_q3_twice.py`, `simulate_option_b_q3_twice.py` | the same for two blocks of $t = \pi/4$ | local |
| `hardware_option_b_q3_pi_over_2.py` | PST at $t = \pi/2$, Option B with $r = 8$, cz basis | hardware (1 job) |
| `hardware_option_b_q3.py` | uniform mixing at $t = \pi/4$, Option B with $r = 4$, cz basis | hardware (1 job) |
| `hardware_q3_fractional.py` | the same two circuits in the fractional basis | hardware (2 jobs) |
| `q3_fractional_survey.py` | device gate counts in both bases and exact noiseless distributions | account |

The Option A scripts synthesize a $256 \times 256$ unitary and are slow to
transpile.

### `hamming_path/`: PST on the Hamming-weighted paths $\mathrm{HP}_n$ (Section 7)

$\mathrm{HP}_n$ is the weighted path on the vertices $0, 1, \ldots, n$ in
which the edge $(k-1, k)$ has weight $\sqrt{k(n-k+1)}$. It is the quotient
of the hypercube $Q_n$ by the distance partition from a vertex, and it has
PST from $0$ to $n$ at $t = \pi/2$.

| Script | What it does | Needs |
|---|---|---|
| `hamming_path_common.py` | Hamiltonian and circuit builders, imported by the other scripts | |
| `demo_hamming_path.py` | both constructions at $t = \pi/4$, $\pi/2$ and two blocks of $\pi/4$, for small $n$ | local |
| `hamming_path_pst_survey.py`, `hamming_path_pst_survey_continued.py` | gate counts for Option A and Option B ($r = 8$), $n \le 12$; the Option A synthesis fails at $n = 10$ | local (slow) |
| `hamming_path_pst_optB_to30.py` | Option B gate counts ($r = 8$) for $n = 2, \ldots, 30$ | local |
| `hamming_path_r_sweep.py` | Trotter-rep sweep for $n \in \{4, 8, 12, 16, 20\}$ and $r \in \{2, 4, 8, 16, 32\}$ | local |
| `hamming_path_pst_r_optimal.py` | $r^*$ for $n = 2, \ldots, 20$ | local |
| `hamming_path_aer_per_vertex_optimal_r.py` | noiseless probabilities of all single-excitation states at $r^*$ (Appendix A) | local |
| `hamming_path_fractional_survey.py` | device gate counts in both bases at $r^*$ | account |
| `hardware_hamming_path_pst_optimal_r.py` | hardware sweep at $r^*$, $n = 2, \ldots, 20$, cz basis | hardware (19 jobs) |
| `hardware_hamming_path_pst_fractional.py` | the same sweep in the fractional basis | hardware (19 jobs) |
| `hardware_hamming_path_pst_optB.py` | $n = 2, 3, 4$ at $r = 8$, for the comparison with PST on $Q_n$ | hardware (3 jobs) |
| `q4_transpile_counts.py` | gate counts for PST on $Q_4$ (16 qubits) on a fake Heron backend (`FakeFez`) | local |
| `hp_pst_chart.py` | bar chart of the target probabilities | local |

### `xn/`: peak state transfer on $X_n$ (Section 8)

$X_n$ is the weighted path on $2n$ vertices of Coutinho, Guo and Schmeits
(see the reference below), with peak state transfer from $w_1$ (qubit 1) to
$w_n$ (qubit $2n-1$) at $t = \pi$.

| Script | What it does | Needs |
|---|---|---|
| `xn_path_common.py` | Hamiltonian and circuit builders, imported by the other scripts | |
| `demo_xn_path.py` | edge weights, spectrum, peak transfer probability and noiseless target probability for $n \le 5$ | local |
| `xn_pst_survey.py` | gate counts for Option A ($n \le 5$) and Option B ($r = 8$) | local (slow) |
| `xn_pst_optB_survey.py` | Option B gate counts ($r = 8$) for $n = 2, \ldots, 20$ | local |
| `xn_pst_r_sweep.py` | Trotter-rep sweep for $n \in \{2, 4, 6, 8, 10\}$ and $r \in \{2, 4, 8, 16, 32\}$ | local |
| `xn_pst_r_optimal.py` | $r^*$ for $n = 2, \ldots, 11$ | local |
| `xn_aer_per_vertex_optimal_r.py` | noiseless probabilities of all single-excitation states at $r^*$ (Appendix B) | local |
| `xn_fractional_survey.py` | device gate counts in both bases at $r^*$ | account |
| `hardware_xn_pst_optimal_r.py` | hardware sweep at $r^*$, $n = 2, \ldots, 11$, cz basis | hardware (10 jobs) |
| `hardware_xn_pst_fractional.py` | the same sweep in the fractional basis | hardware (10 jobs) |
| `xn_pst_chart.py` | bar chart of the target probabilities | local |

### `swap_chain/`: SWAP-chain baseline (Section 9)

| Script | What it does | Needs |
|---|---|---|
| `swap_chain_baseline.py` | the SWAP chain on $n = 2, \ldots, 21$ qubits on Aer | local |
| `hardware_swap_chain_baseline.py` | the SWAP chain on `ibm_kingston`; a dry run unless given `--submit` | account; hardware (20 jobs) with `--submit` |
| `swap_chain_figure.py` | figure of the circuit for $n = 10$ | local |
| `transfer_distance_chart.py` | bar chart comparing the SWAP chain, $\mathrm{HP}_n$ and $X_n$ at equal transfer distance | local |

### Top level

| Script | What it does | Needs |
|---|---|---|
| `save_ibm_account.py` | stores IBM Quantum credentials | |
| `retrieve_job_full.py` | prints the full output distribution of a finished job: `python retrieve_job_full.py <JOB_ID>` | account |

## Notes

- With the versions in `requirements.txt`, the local scripts give the
  two-qubit gate counts and noiseless probabilities of the paper. Depth
  and total gate counts depend on the Qiskit version and can differ from
  the paper.
- The chart scripts contain the plotted values in a list at the top of the
  file, copied from the output of the Aer and hardware scripts.
- The cz-basis hardware scripts for $C_4$ and $Q_3$, and
  `hardware_hamming_path_pst_optB.py`, submit to the least busy backend
  (`service.least_busy(...)`); the runs in the paper were on
  `ibm_kingston`. To fix the device, replace that call with
  `service.backend("ibm_kingston")`. The other hardware scripts use
  `ibm_kingston` directly.
- Hardware results depend on the calibration of the device at the time of
  the run.
- The drawings of the logical circuits in Section 5 of the paper were not
  made with these scripts.

## Reference

G. Coutinho, K. Guo and V. Schmeits, *Peak state transfer in continuous
quantum walks*, arXiv:2505.11986 (2025).

## License

The code is released under the MIT License; see `LICENSE`.
