# model_benchmarking

Tools for benchmarking dark QCD models with emerging-jet signatures.
Supports two mediator topologies:

- **T-channel** — mediator X couples SM down-type quarks to dark quarks via a Yukawa-like κ matrix
- **S-channel** — massive Z' vector mediator with flavor-dependent dark-quark charges

For each model the package computes dark pion decay widths, lifetimes, and
branching ratios, and provides plotting scripts for parameter-space visualisation.
Card generation for MadGraph/Pythia is currently supported for the **t-channel
model only**.

## Installation

Clone the repo (with submodules) and install the `dark_pion_widths` package:

```bash
git clone --recurse-submodules <repo-url>
cd model_benchmarking
pip install -e .
```

**Dependencies:** `numpy`, `matplotlib` (matplotlib only needed for plotting).

---

## Inspecting a benchmark model

`summarize_models.py` prints a formatted summary of every benchmark model
configuration defined in `plotting/model_config.py`:

```bash
python summarize_models.py
```

You can also call `print_summary()` on any model instance directly (see the
API section below).

---

## Generating cards (t-channel only)

`write_cards.py` is the entry point for MadGraph/Pythia card generation. Given
a set of t-channel model parameters it:
- solves for κ (or computes c·τ if κ is provided directly)
- writes a MadGraph `param_card.dat` and `run_card.dat`
- writes a Pythia `pythia_card.dat` with pion masses, lifetimes, and branching ratios
- writes a `launch.sh` script that runs MadGraph in batch mode

The c·τ target always refers to a single **reference species** (T15 by default).
Off-diagonal pion lifetimes are computed independently from the solved κ.

### κ matrix modes

The shape of the κ coupling matrix is controlled by `--kappa-mode`:

| Mode | Description |
|---|---|
| `universal` | All entries equal to κ (default) |
| `diagonal` | Only diagonal entries κ₁₁, κ₂₂, κ₃₃ non-zero |
| `down_only` | Only first column κ₁₁, κ₂₁, κ₃₁ non-zero (d-quark coupling only) |

### Parameters

| Argument | Required | Default | Description |
|---|---|---|---|
| `--mXd` | yes | — | Mediator mass [GeV] |
| `--mPiD` | yes | — | Dark pion mass [GeV] |
| `--ctau` | one of | — | Target c·τ [mm] for the reference species; solves for κ |
| `--kappa` | one of | — | Coupling κ; computes c·τ from it |
| `--fD` | no | `mPiD` | Dark decay constant [GeV] |
| `--LambdaD` | no | `mPiD` | Dark confinement scale [GeV] |
| `--mDarkQ` | no | `2 × mPiD` | Dark quark mass [GeV] |
| `--mRhoD` | no | `4 × mPiD` | Dark rho mass [GeV] |
| `--species` | no | `T15` | Reference pion species for c·τ target |
| `--kappa-mode` | no | `universal` | κ matrix structure (see above) |
| `--nevents` | no | `10000` | Number of events in run card |
| `--sqrts` | no | `13600` | Centre-of-mass energy [GeV] |
| `--lhe` | no | `events.lhe` | LHE filename written into Pythia card |
| `--process-dir` | no | placeholder | MadGraph process directory for `launch.sh` |
| `--mg5-exe` | no | `mg5_aMC` | MadGraph executable |
| `--output` | no | auto | Output directory |

At least one of `--ctau` or `--kappa` must be given. If both are given,
`--kappa` takes precedence and the actual c·τ is computed from it.

### Examples

```bash
# Solve for kappa given a target lifetime
python write_cards.py --mXd 1500 --mPiD 10 --ctau 100

# Use kappa directly, compute lifetime
python write_cards.py --mXd 2000 --mPiD 10 --kappa 0.5

# Diagonal kappa matrix, custom output directory
python write_cards.py --mXd 2000 --mPiD 15 --ctau 50 \
    --kappa-mode diagonal \
    --fD 15 --LambdaD 4 --nevents 50000 \
    --process-dir mg5_output/mediator_pair_down \
    --output cards/my_point
```

Output cards are written to `cards/mXd{mXd}_mPiD{mPiD}_ctau{ctau}/` by
default (or `_kappa{kappa}` if only `--kappa` is given).

---

## Running MadGraph

### Step 1 — Generate the process (once)

Ready-made proc cards are in `proc_cards/`. Run the one you want:

```bash
mg5_aMC proc_cards/mediator_pair.txt
# → mg5_output/mediator_pair/

mg5_aMC proc_cards/dark_quark_pair_multijet.txt
# → mg5_output/dark_quark_pair_multijet/
```

The dark quark multijet card generates `p p > qd qd`, `p p > qd qd j`, and
`p p > qd qd j j` as a combined process.

This compiles the matrix elements once — reuse the same directory for all
parameter points.

### Step 2 — Launch a run (per parameter point)

Pass `--process-dir` when generating cards, then run the launch script:

```bash
python write_cards.py --mXd 1500 --mPiD 10 --ctau 100 \
    --process-dir mg5_output/mediator_pair_down

bash cards/mXd1500_mPiD10_ctau100/launch.sh
```

`launch.sh` copies `param_card.dat` and `run_card.dat` into the process
directory and calls `echo "launch -f" | mg5_aMC <process_dir>`. The `-f` flag
skips all interactive prompts. Events land in
`mg5_output/mediator_pair_down/Events/`.

You can also override the process directory and executable at runtime:

```bash
bash launch.sh /path/to/process_dir /path/to/mg5_aMC
```

---

## Running Pythia

Point Pythia at the LHE file produced by MadGraph:

```bash
python write_cards.py ... --lhe mg5_output/mediator_pair_down/Events/run_01/unweighted_events.lhe.gz
```

Then pass `pythia_card.dat` to your Pythia driver. The card sets all dark
sector masses, lifetimes, branching ratios, and HiddenValley parameters.

---

## Submitting a parameter grid (Slurm)

`submit_grid.py` automates card generation and Slurm job submission for a full
Cartesian grid of model parameters. It requires no hardcoded cluster settings —
all Slurm resource options are supplied on the command line.

### Prerequisites

The compiled MadGraph process directory must already exist (see Step 1 above).

### Example

```bash
python submit_grid.py \
    --mXd 1000 2000 3000 4000 \
    --mPiD 5 10 20 \
    --ctau 1 10 100 \
    --process-dir mg5_output/mediator_pair_down \
    --env-setup "source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh" \
    --partition cpu \
    --time 04:00:00 \
    --account myproject \
    --nevents 10000
```

This submits 4 × 3 × 3 = 36 Slurm jobs. Use `--kappa` instead of `--ctau` to
specify the coupling directly.

Always preview job scripts before submitting:

```bash
python submit_grid.py ... --dry-run
```

### All options

| Option | Description |
|---|---|
| `--mXd` | Mediator mass values [GeV] (space-separated list) |
| `--mPiD` | Dark pion mass values [GeV] |
| `--ctau` | Target c·τ values [mm] |
| `--kappa` | Coupling κ values (mutually exclusive with `--ctau`) |
| `--process-dir` | Compiled MadGraph process directory |
| `--env-setup` | Shell command to set up the environment in each job |
| `--partition` | Slurm partition / queue |
| `--account` | Slurm account / project |
| `--time` | Wall-clock time limit (e.g. `04:00:00`) |
| `--mem` | Memory per node (e.g. `4G`) |
| `--cpus-per-task` | CPUs per task |
| `--constraint` | Node feature constraint (e.g. `el9`) |
| `--sbatch-opt` | Any additional `#SBATCH` directive (repeatable) |
| `--runs-dir` | Directory for per-point run dirs (default: `runs/`) |
| `--logs-dir` | Directory for Slurm stdout/stderr logs (default: `logs/`) |
| `--cards-base` | Base directory for generated cards (default: `cards/`) |
| `--nevents` | Events per job (default: 10000) |
| `--sqrts` | Centre-of-mass energy [GeV] (default: 13600) |
| `--dry-run` | Print job scripts without submitting |

Events land in `runs/<point>/Events/<point>/`.
Slurm logs are written to `logs/<point>.out` and `logs/<point>.err`.

---

## `dark_pion_widths` API

Both model classes inherit from `DarkPionModelBase` and share a common
interface. The base class holds all common parameters (`fD`, `m_piD`, `Nf`,
`Nd`, `Nc`, `sm_quarks`) and provides the `width_to_ctau_mm` utility.

### T-channel model

```python
from dark_pion_widths import DarkPionTChannelModel

model = DarkPionTChannelModel(
    fD=10.0,          # dark decay constant [GeV]
    m_piD=10.0,       # dark pion mass [GeV]
    m_X=2000.0,       # mediator mass [GeV]
    kappa=0.5,        # coupling (complex)
    Nf=4,             # dark quark flavors
    kappa_mode="universal",  # "universal" | "diagonal" | "down_only"
    Nc=3,             # SM colours
    Nd=3,             # dark colours
)
```

### S-channel model

```python
from dark_pion_widths import DarkPionSChannelModel

model = DarkPionSChannelModel(
    Nf=3,                      # dark quark flavors
    Nd=3,                      # dark colours
    dark_charges=[1.0, 2.0, 3.0],  # U(1)' charge per dark flavor
    fD=10.0,                   # dark decay constant [GeV]
    m_piD=10.0,                # dark pion mass [GeV]
    m_Zp=1500.0,               # Z' mass [GeV]
    g_qd=0.3,                  # Z' coupling to dark quarks
    g_q=0.3,                   # Z' coupling to SM quarks
    Nc=3,                      # SM colours
)
```

### Common interface

Both classes expose the same methods:

```python
# Compute all pion widths at once
results = model.compute_all()
# returns: {
#   "off_diagonal": {(alpha, beta): {...}},  # decaying pairs only
#   "diagonal":     {b: {...}},              # decaying diagonal pions only
#   "quark_labels": [str, ...],
# }

# Per-pion access
r = model.compute_diagonal_pion(b)        # b = generator index (int)
r = model.compute_off_diagonal_pion(a, b) # (a, b) = flavor pair indices

# c·τ convenience accessors [mm]
model.ctau_diag_mm(14)          # T15 (Nf=4)
model.ctau_diag_mm(13)          # T8
model.ctau_off_diag_mm(0, 1)    # π^(0,1)

# Iterate over decaying pions
for (a, b), r in results["off_diagonal"].items():
    print(f"pi^({a},{b}): ctau = {r['ctau_mm']:.3e} mm, total Gamma = {r['total']:.3e} GeV")

for b, r in results["diagonal"].items():
    print(f"pi^T{b}: ctau = {r['ctau_mm']:.3e} mm")

# Formatted summary (prints to stdout)
model.print_summary()
```

### Generator index conventions

Dark pions are the Nf²−1 generators of SU(Nf). The ordering used here is:
- **Off-diagonal**: pairs (j,k) with j < k, each appearing twice (symmetric/antisymmetric). For Nf=4: indices 0–11.
- **Diagonal**: T3, T8, T15, … For Nf=4: indices 12, 13, 14.

The helper `get_generator_label(Nf, a)` maps a generator index to its label
(`(j,k)` for off-diagonal, `T3`/`T8`/`T15` for diagonal).

---

## Plotting

All plotting scripts live in `plotting/` and are run as standalone scripts.
Benchmark model configurations are defined in `plotting/model_config.py`.
Each script loops over all configured models and writes one PDF per model per
plot type.

### c·τ vs model parameters

```bash
python plotting/dark_pion_lifetime.py
```

Produces 1D lifetime scans vs m_πD, mediator mass, coupling, and fD, plus 2D
scans of m_πD and coupling vs mediator mass. Output filenames follow the
pattern `dark_pion_lifetime_<scan>_<model_tag>.pdf`.

### Iso-lifetime contours

```bash
python plotting/ctau_contours.py
```

Iso-c·τ contours in the (m_πD, m_mediator), (m_πD, coupling), and
(m_mediator, coupling) planes, plus an overlay across all pion species.
Output: `ctau_contours_<plane>_<model_tag>.pdf`.

### Branching ratios

```bash
python plotting/branching_ratios.py
```

Branching ratios vs m_πD for each decaying pion species in each benchmark
model. Output: `dark_pion_branching_ratios_<model_tag>_<species>.pdf`.

### Coupling solutions

```bash
python plotting/coupling_solutions.py
```

Required coupling κ (or g_qd) as a function of (m_πD, m_mediator) for a
set of target c·τ values, as both heatmaps and line plots.
Output: `coupling_solutions_heatmap_<model_tag>.pdf`,
`coupling_solutions_lines_<model_tag>.pdf`.

### Cross section vs mediator mass

```bash
# From a Slurm grid output directory
python plotting/cross_section.py --runs-dir runs/

# From a single MadGraph process directory
python plotting/cross_section.py --process-dir mg5_output/mediator_pair_down
```

Options: `--output FILE`, `--no-group` (plot all runs as one series instead
of grouping by κ).
