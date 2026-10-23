# HepMC analysis for the dark-QCD / emerging-jets samples

Streams the Pythia HepMC3 files of the grid, writes compact per-event and
per-object tables, and overlays normalised distributions to compare event
topologies across samples and parameters.

| File | Purpose |
|---|---|
| `env.sh` | source first: sets `$EJ_BASE`, `$EJ_OUT`, `$EJ_PY` (site overrides in `env.local.sh`) |
| `requirements.txt` | Python packages needed by the analysis |
| `common.py` | PDG-ID classes, grid discovery, card/log parsing, kinematics helpers |
| `extract.py` | HepMC file -> `events/pions/jets/mediators.parquet` + `meta.json` |
| `run_grid.py` | run `extract.py` over the grid: local thread pool or a Slurm array |
| `plot.py` | overlay any selection of extracted samples (normalised histograms) |
| `validate.py` | proper-ctau vs card tau0 check, grid summary table, ctau plots |
| `plot_benchmarks.py` | overlay arbitrary extracted directories (points outside the grid) |

## Environment

Python >= 3.9 with the packages in `requirements.txt` (`pyhepmc`, scikit-hep
`fastjet`, `particle`, plus numpy/pandas/pyarrow/matplotlib/scipy), e.g.

```bash
python3 -m venv .venv-ana && .venv-ana/bin/pip install -r analysis/requirements.txt
```

Then

```bash
export EJ_BASE=/path/to/production   # holds runs/<sample>/<point>/ and cards/<sample>/<point>/
export EJ_PY=$PWD/.venv-ana/bin/python
source analysis/env.sh               # EJ_OUT defaults to $EJ_BASE/analysis_outputs
cd analysis
```

The grid layout matches `submit_grid.py` run once per sample with
`--runs-dir runs/<sample> --cards-base cards/<sample>` and `--pythia-exe`, i.e.
`$EJ_BASE/runs/<sample>/<point>/Events/<point>/<point>.hepmc.gz`. Anything else
can be processed with `extract.py --input <file> --outdir <dir>`. Site-specific
settings (paths, interpreter, cache locations) can be put in
`analysis/env.local.sh`, which `env.sh` sources if present and git ignores.

## Running

```bash
# one file (~1 min / 1000 events, <1 GB RAM)
$EJ_PY extract.py --sample mediator_onshell --point mXd2000_mPiD12_ctau10
$EJ_PY extract.py --input file.hepmc.gz --outdir /some/dir --max-events 100   # ad hoc

# the grid (skips points that already have meta.json unless --force)
$EJ_PY run_grid.py --mode slurm --partition <queue>   # Slurm array, logs in $EJ_OUT/logs/slurm
$EJ_PY run_grid.py --mode local -j 4 --samples onshell --mX 2000

# checks + summary table ($EJ_OUT/validation/)
$EJ_PY validate.py

# plots ($EJ_OUT/plots/<tag>/all.pdf + one png per variable)
$EJ_PY plot.py --preset standard                   # vary ctau / mX / mPiD, on-shell vs merged
$EJ_PY plot.py --samples onshell --mX 2000 --mPiD 12 --tag my_ctau_scan
$EJ_PY plot.py --samples onshell merged --mX 2000 --mPiD 12 --ctau 10 \
               --vars met n_pi pions:Lxy jets:alpha --jet-cut "pt > 300"
$EJ_PY plot.py --list-vars
```

Selections take sample short names (`onshell`, `merged`) and any lists of
`--mX/--mPiD/--ctau`; every extracted combination is overlaid (max 8 per plot).
Histograms are normalised to unit area; over/underflow is put in the edge bins.
Outputs live in `$EJ_OUT/features/<sample>/<point>/`; load them with pandas
(`pd.read_parquet`) for anything not covered by `plot.py`.

## Definitions

* **Final state**: HepMC status 1. **Invisible**: final neutrinos (12/14/16) and
  any final dark-sector particle (|pid| in [4900000, 4999999]; in practice the
  stable pions 4900411/421/431/441). **Visible**: all other final particles.
* **Last copy**: a particle with no same-PID daughter (strips shower/recoil copies).
  Production position is taken from the first copy of the chain.
* Positions are HepMC vertex positions (mm); vertices without an explicit
  position inherit it from their incoming particle (HepMC3 rule). The PV is at
  the origin (no beam spot in these samples).
* **Dark-pion descendants**: everything downstream of the decay vertex of a
  decayed dark pion (graph traversal); used for `from_pi` flags.
* **Jets**: anti-kT R = 1.0 (FastJet) on all visible final particles, stored if
  pT > 50 GeV and |eta| < 4.5. **Tracks**: charged visible final particles with
  pT > 1 GeV, |eta| < 2.5 and production-vertex Lxy < 300 mm.

### `events.parquet` (one row per event)

| column | meaning |
|---|---|
| `weight` | event weight (pb, Pythia/MG5 convention: constant for these samples) |
| `n_pi`, `n_pi_decaying`, `n_pi_stable` | dark pions (last copies): total / decaying species (111,221,331,211,311,321) / stable species (411,421,431,441) |
| `n_pi_decayed` | decaying-species pions that actually decayed (should equal `n_pi_decaying`) |
| `n_pi_diag`, `n_pi_offdiag`, `n_pi_eta4` | split by species type |
| `frac_pi_stable` | n_stable / n_pi |
| `n_rho`, `n_dark_hadrons` | dark rhos (last copies), pions + rhos |
| `pi_lxy_mean/median/min/max` | transverse decay length of decayed dark pions (mm, relative to production) |
| `n_pi_lxy_lt1, _1_10, _10_300, _300_3000, _gt3000` | number of decayed pions per Lxy bin (mm) |
| `n_pi_outside_calo` | pions decaying beyond r = 4 m or \|z\| = 6 m |
| `pi_pt_mean`, `pi_pt_lead` | dark-pion pT mean / max |
| `n_dq_hard`, `n_dq_fromX`, `n_dq_ME` | status-23 dark quarks: all / from X decay / from the matrix element |
| `n_sm_parton_hard` | status-23 SM partons not from X decay (ME extra jets in the merged sample) |
| `n_X` | **mediators X at their last copy** (i.e. "final" X, one per physical X) |
| `n_X_records`, `n_X_hard` | all X records incl. shower copies; X with status 22 |
| `X1_pt/eta/m`, `X2_pt/eta` | leading / subleading X kinematics |
| `n_bhad`, `n_bhad_from_pi` | weakly-decaying b hadrons (no b-hadron daughter), all / from dark-pion decays |
| `f_vis_pt_darkpi`, `f_vis_e_darkpi` | fraction of visible scalar-pT (energy) sum from dark-pion descendants |
| `met`, `met_phi` | true MET: \|vector pT sum\| of all invisible final particles |
| `met_sm`, `met_dark` | same for neutrinos only / dark particles only |
| `met_det`, `met_det_phi` | detector-like MET: invisible + visible with \|eta\|>4.9 + visible products of pions decaying outside r=4 m / \|z\|=6 m |
| `n_trk`, `n_trk_disp`, `f_trk_disp` | tracks; displaced tracks (origin 2 < Lxy < 300 mm); fraction |
| `n_jet50`, `n_jet200`, `n_jet200_c` | jets pT>50; pT>200; pT>200 and \|eta\|<2.5 |
| `ht50`, `ht200` | scalar pT sum of jets with pT > 50 / 200 GeV |
| `n_ej_proxy` | jets with pT>200, \|eta\|<2.5 and alpha < 0.25 (incl. alpha = -1, no tracks) |
| `j1_*`, `j2_*` | leading/subleading jet: pt, eta, phi, m, alpha, n_trk, n_trk_disp, frac_pt_darkpi, frac_pt_disp1, n_darkpi_dR |
| `dR_j1j2`, `dphi_j1j2`, `mjj` | leading-pair geometry and invariant mass |
| `met_over_ht`, `met_det_over_ht` | MET / ht50 |
| `dphi_met_jclosest`, `dphi_met_j1`, `dphi_metdet_jclosest` | \|dphi\| between MET and the closest jet (pT>50) / leading jet |

### `pions.parquet` (one row per dark pion, last copy)

`pid, apid, status, decaying_species, decayed, pt, eta, phi, p, e, m, from_rho,
parent_apid, prod_{x,y,z}`; for decayed pions also `dec_{x,y,z}`, `L` (3D lab
decay length), `Lxy`, `Lz`, `dec_r`, `dec_absz`, `ctau = L m / |p|` (proper
decay length, mm), `ctau_t = Δt m / E` (same from vertex times; cross-check),
`betagamma`, `n_vis_desc`, `vis_desc_pt`, `n_bhad_desc`.

### `jets.parquet` (one row per jet)

`rank, pt, eta, phi, m, n_const, n_charged, n_trk, n_trk_disp`,
`alpha` = pT fraction of the jet's tracks whose origin is within 1 mm (3D) of
the PV (-1 if the jet has no tracks; small alpha = emerging),
`frac_pt_disp1/10` (constituent-pT fraction with origin Lxy > 1/10 mm),
`frac_pt_darkpi` (from dark-pion descendants), `frac_pt_charged`
(fractions are of the scalar constituent-pT sum), `n_darkpi_dR` /
`n_stablepi_dR` (decaying / stable dark pions within ΔR < 1).

### `mediators.parquet` (one row per X, last copy)

`pid, status, pt, eta, phi, m, first_status`, decay products: `q_pid, q_pt,
q_eta, q_phi` (SM quark), `dq_pid, dq_pt, dq_eta, dq_phi` (dark quark), `dR_q_dq`.

### `meta.json`

point parameters, `n_events`, `xsec_pb` (GenCrossSection of the last event,
i.e. Pythia's final estimate incl. MLM vetoes), `xsec_mg5_pb` (banner),
`sigmaGen_fb` and accepted/read counts (pythia.log), the card header, `tau0_mm`
and `m0` per PDG ID, the analysis configuration and the processing time.

## Caveats

* Truth-level only; tracks/alpha/MET_det are crude proxies (no smearing, no efficiencies).
* The extractor assumes contiguous particle ids (true for Pythia output); it stops otherwise.
* alpha uses a 1 mm (3D) prompt cut, so even ctau = 1 mm pions give small alpha;
  tracks are dropped beyond Lxy = 300 mm, so very displaced jets get alpha = -1 or 1.
* For a wide mediator the on-shell sample (decay chains, `bwcutoff` x width
  window) can be dominated by far off-shell X; check `mediators:m` before
  treating such points as resonant.
* MadGraph can stop before reaching `nevents` when the unweighting efficiency
  is low (merged sample at large m_X); check `n_events` in `meta.json`.
* Stable dark pions are produced as `isVisible = off` final particles and
  therefore count as MET. Which species are stable is set by the Pythia card;
  note that diagonal pions are also produced in dark-rho decays.
