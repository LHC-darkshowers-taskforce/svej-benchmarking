# svej-benchmarking

Tools for benchmarking dark QCD models with a t-channel mediator.
Computes dark pion widths, lifetimes, branching ratios, and solves for the
coupling κ, then generates MadGraph and Pythia cards ready for event generation.

## Installation

Clone the repo (with submodules) and install the `dark_pion_widths` package:

```bash
git clone --recurse-submodules <repo-url>
cd model_benchmarking
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

Write a `proc_card.txt`:

```
import model models/t-channel_dark_QCD/darkQCD_fv_down
generate p p > x x~
output mg5_output/mediator_pair_down
```

Run it:

```bash
mg5_aMC proc_card.txt
```

This compiles the matrix elements into `mg5_output/mediator_pair_down/`. Only
needs to be done once — reuse the same directory for all parameter points.

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

Standalone scripts for visualising the parameter space. All write PDFs to the
current directory.

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
