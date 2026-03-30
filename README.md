# svej-benchmarking

Tools for benchmarking dark QCD models with a t-channel mediator.
Computes dark pion widths, lifetimes, branching ratios, and solves for the
coupling κ, then generates MadGraph and Pythia cards ready for event generation.

## Installation

Clone the repo (with submodules) and install the `dark_pion_widths` package:

```bash
git clone --recurse-submodules <repo-url>
cd svej-benchmarking
pip install -e .
```

**Dependencies:** `numpy`, `matplotlib` (matplotlib only needed for plotting).

## Generating cards

`write_cards.py` is the main entry point. Given a set of model parameters it:
- solves for κ (or computes c·τ if κ is provided directly)
- writes a MadGraph `param_card.dat` and `run_card.dat`
- writes a Pythia `pythia_card.dat` with pion masses, lifetimes, and branching ratios
- writes a `launch.sh` script that runs MadGraph in batch mode

The c·τ target always refers to the **diagonal** dark pions (T15 by default).
Off-diagonal pion lifetimes are computed independently from the solved κ using
the correct SU(4) matrix elements — they will generally differ from the diagonal value.

### Parameters

| Argument | Required | Default | Description |
|---|---|---|---|
| `--mXd` | yes | — | Mediator mass [GeV] |
| `--mPiD` | yes | — | Dark pion mass [GeV] |
| `--ctau` | one of | — | Target diagonal pion c·τ [mm]; solves for κ |
| `--kappa` | one of | — | Coupling κ; computes c·τ from it |
| `--fD` | no | `mPiD` | Dark decay constant [GeV] |
| `--LambdaD` | no | `mPiD` | Dark confinement scale [GeV] |
| `--mDarkQ` | no | `2 × mPiD` | Dark quark mass [GeV] |
| `--mRhoD` | no | `4 × mPiD` | Dark rho mass [GeV] |
| `--species` | no | `T15` | Reference pion species for c·τ target |
| `--nevents` | no | `10000` | Number of events in run card |
| `--sqrts` | no | `13600` | Centre-of-mass energy [GeV] |
| `--lhe` | no | `events.lhe` | LHE filename written into Pythia card |
| `--process-dir` | no | placeholder | MadGraph process directory for `launch.sh` |
| `--mg5-exe` | no | `mg5_aMC` | MadGraph executable |
| `--output` | no | auto | Output directory |

At least one of `--ctau` or `--kappa` must be given. If both are given, `--kappa`
takes precedence and the actual c·τ is computed from it.

### Examples

```bash
# Solve for kappa given a target lifetime
python write_cards.py --mXd 1500 --mPiD 10 --ctau 100

# Use kappa directly, compute lifetime
python write_cards.py --mXd 2000 --mPiD 10 --kappa 0.5

# Full set of parameters, custom output directory
python write_cards.py --mXd 2000 --mPiD 15 --ctau 50 \
    --fD 15 --LambdaD 4 --nevents 50000 \
    --process-dir mg5_output/mediator_pair_down \
    --output cards/my_point
```

Output cards are written to `cards/mXd{mXd}_mPiD{mPiD}_ctau{ctau}/` by default
(or `_kappa{kappa}` if only `--kappa` is given).

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
`p p > qd qd j j` as a combined process (`qd` is a multiparticle label for
`dDark1 dDark1~ dDark2 dDark2~ dDark3 dDark3~`).

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
skips all interactive prompts. Events land in `mg5_output/mediator_pair_down/Events/`.

You can also override the process directory and executable at runtime:

```bash
bash launch.sh /path/to/process_dir /path/to/mg5_aMC
```

## Running Pythia

Point Pythia at the LHE file produced by MadGraph:

```bash
python write_cards.py ... --lhe mg5_output/mediator_pair_down/Events/run_01/unweighted_events.lhe.gz
```

Then pass `pythia_card.dat` to your Pythia driver. The card sets all dark
sector masses, lifetimes, branching ratios, and HiddenValley parameters.

## Submitting a parameter grid (Slurm)

`submit_grid.py` automates card generation and Slurm job submission for a full
Cartesian grid of model parameters.  It requires no hardcoded cluster settings —
all Slurm resource options are supplied on the command line, so the script works
on any cluster.

### Prerequisites

The compiled MadGraph process directory must already exist (see Step 1 above).
Generate it once using a proc card from `proc_cards/`:

```bash
mg5_aMC proc_cards/mediator_pair.txt
```

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

This submits 4 × 3 × 3 = 36 Slurm jobs.  Use `--kappa` instead of `--ctau` to
specify the coupling directly.

### Dry run

Always preview the job scripts before submitting:

```bash
python submit_grid.py ... --dry-run
```

### All options

| Option | Description |
|---|---|
| `--mXd` | Mediator mass values [GeV] (space-separated list) |
| `--mPiD` | Dark pion mass values [GeV] |
| `--ctau` | Target diagonal pion c·τ values [mm] |
| `--kappa` | Coupling κ values (mutually exclusive with `--ctau`) |
| `--process-dir` | Compiled MadGraph process directory |
| `--env-setup` | Shell command to set up the environment in each job |
| `--partition` | Slurm partition / queue |
| `--account` | Slurm account / project |
| `--time` | Wall-clock time limit (e.g. `04:00:00`) |
| `--mem` | Memory per node (e.g. `4G`) |
| `--cpus-per-task` | CPUs per task |
| `--constraint` | Node feature constraint (e.g. `el9`) |
| `--sbatch-opt` | Any additional `#SBATCH` directive, e.g. `--sbatch-opt='--nodes=1'` (repeatable) |
| `--runs-dir` | Directory for per-point run dirs (default: `runs/`) |
| `--logs-dir` | Directory for Slurm stdout/stderr logs (default: `logs/`) |
| `--cards-base` | Base directory for generated cards (default: `cards/`) |
| `--nevents` | Events per job (default: 10000) |
| `--sqrts` | Centre-of-mass energy [GeV] (default: 13600) |
| `--dry-run` | Print job scripts without submitting |

### Output

Each job hard-link-copies the compiled process directory into
`runs/<point>/` for isolation (no race condition between parallel jobs), then
copies the generated cards and launches MadGraph.

Events land in:

```
runs/<point>/Events/<point>/
```

Slurm logs are written to `logs/<point>.out` and `logs/<point>.err`.

---

## `dark_pion_widths` API

The `dark_pion_widths` package can also be used directly:

```python
from dark_pion_widths import DarkPionModel

model = DarkPionModel(fD=10.0, m_piD=10.0, m_X=2000.0, kappa=0.5)
results = model.compute_all()

# c·τ of diagonal pions
print(model.ctau_diag_mm(14))   # T15
print(model.ctau_diag_mm(7))    # T8
print(model.ctau_diag_mm(2))    # T3

# c·τ of off-diagonal pions
print(model.ctau_off_diag_mm(0, 1))

# Full results: widths, branching ratios, lifetimes
for (a, b), r in results['off_diagonal'].items():
    print(f"pi^({a},{b}): ctau={r['ctau_mm']:.3e} mm")
```

---

## Plotting

### Cross section vs mediator mass

`plot_xsec.py` extracts cross sections from MadGraph output and plots σ vs m_Xd,
grouped by κ.

**Slurm grid output** (produced by `submit_grid.py`):

```bash
python plotting/plot_xsec.py --runs-dir runs/
# Output: xsec.pdf
```

**Single process directory**:

```bash
python plotting/plot_xsec.py --process-dir mg5_output/mediator_pair_down
# Output: xsec.pdf
```

Options:

```
--runs-dir DIR       Slurm grid output directory (one subdir per parameter point)
--process-dir DIR    Single MadGraph process directory (Events/run_*/ layout)
--output FILE        Output filename (default: xsec.pdf)
--no-group           Plot all runs as one series instead of grouping by κ
```

### Parameter-space visualisation

Standalone scripts for visualising the dark pion parameter space. All write PDFs
to the current directory.

```bash
# c·τ vs model parameters (1D and 2D scans)
python plotting/plot_dark_pion_lifetime.py
# Output: dark_pion_lifetime_1d.pdf, dark_pion_lifetime_2d.pdf,
#         dark_pion_lifetime_mpiD_vs_mX.pdf, dark_pion_lifetime_kappa_vs_mX.pdf

# Iso-lifetime contours in 2D parameter planes
python plotting/plot_ctau_contours.py
# Output: ctau_contours_mpiD_mX.pdf, ctau_contours_mpiD_kappa.pdf,
#         ctau_contours_mX_kappa.pdf, ctau_contours_kappa_overlay.pdf

# Branching ratios vs m_piD
python plotting/plot_branching_ratios.py
# Output: dark_pion_branching_ratios.pdf

# κ solutions as heatmap and line plots
python plotting/plot_kappa_solutions.py
# Output: kappa_solutions_heatmap.pdf, kappa_solutions_lines.pdf
```
