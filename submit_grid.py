#!/usr/bin/env python3
"""
submit_grid.py
--------------
Generate cards and submit Slurm jobs for a grid of dark QCD parameter points.

For each point in the Cartesian product of the supplied parameter values:
  1. Calls write_cards.py to generate cards in cards/<point>/
  2. Submits a Slurm job that:
       - Sets up the environment (--env-setup)
       - Hard-link-copies the compiled MadGraph process directory so that
         parallel jobs are fully isolated (no card-overwrite race condition)
       - Copies the generated cards into the isolated run directory
       - Launches MadGraph in batch mode

The process directory (--process-dir) must already exist and contain compiled
matrix elements.  Generate it once with:

    mg5_aMC proc_card.txt

All Slurm resource parameters must be supplied explicitly — nothing is
hardcoded, so the script works on any cluster.

Usage
-----
    python submit_grid.py \\
        --mXd 1000 2000 3000 4000 \\
        --mPiD 5 10 20 \\
        --ctau 1 10 100 \\
        --process-dir mg5_output/mediator_pair_down \\
        --env-setup "source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh" \\
        --partition cpu \\
        --time 04:00:00 \\
        --account myproject \\
        --nevents 10000

Use --dry-run to print jobs without submitting.
"""

import argparse
import itertools
import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Slurm job script builder
# ---------------------------------------------------------------------------

def build_job_script(
    point_name: str,
    cards_dir: str,
    process_dir: str,
    runs_dir: str,
    logs_dir: str,
    mg5_exe: str,
    env_setup: str,
    sbatch_opts: dict,
    extra_sbatch: list[str],
) -> str:
    """
    Build a Slurm batch script for one parameter point.

    The job hard-link-copies the compiled process directory so each run is
    fully isolated, then launches MadGraph in batch mode.
    """
    abs_cards   = os.path.abspath(cards_dir)
    abs_process = os.path.abspath(process_dir)
    abs_runs    = os.path.abspath(runs_dir)
    abs_logs    = os.path.abspath(logs_dir)
    run_dir     = os.path.join(abs_runs, point_name)

    lines = ["#!/usr/bin/env bash"]

    # #SBATCH directives
    lines.append(f"#SBATCH --job-name=svej_{point_name}")
    lines.append(f"#SBATCH --output={abs_logs}/{point_name}.out")
    lines.append(f"#SBATCH --error={abs_logs}/{point_name}.err")
    for key, val in sbatch_opts.items():
        if val is not None:
            lines.append(f"#SBATCH --{key}={val}")
    for opt in extra_sbatch:
        lines.append(f"#SBATCH {opt}")

    lines.append("")
    #lines.append("set -euo pipefail")
    lines.append("")

    # Environment setup
    if env_setup:
        lines.append("# Environment setup")
        lines.append(env_setup)
        lines.append("")

    # Isolated run directory (full copy, works across filesystems)
    lines += [
        "# Full copy of compiled process directory — isolates Cards/ and Events/ per job.",
        f'if [[ -d "{run_dir}" ]]; then',
        f'    echo "Run directory already exists: {run_dir} — skipping copy."',
        f'else',
        f'    echo "Creating isolated run directory: {run_dir}"',
        f'    cp -r "{abs_process}" "{run_dir}"',
        f'fi',
        "",
        "# Copy cards for this parameter point",
        f'cp "{abs_cards}/param_card.dat" "{run_dir}/Cards/param_card.dat"',
        f'cp "{abs_cards}/run_card.dat"   "{run_dir}/Cards/run_card.dat"',
        "",
        "# Launch MadGraph in batch mode with a unique run name",
        f'echo "Launching MadGraph for {point_name}"',
        f'cd "{run_dir}"',
        f'echo "launch . -f -n {point_name}" | {mg5_exe}',
        "",
        f'echo "Done. Events are in {run_dir}/Events/{point_name}/"',
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Submit a grid of dark QCD parameter points to Slurm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Grid parameters ---
    grid = p.add_argument_group("grid parameters")
    grid.add_argument("--mXd", type=float, nargs="+", required=True,
                      metavar="GEV", help="Mediator mass values [GeV]")
    grid.add_argument("--mPiD", type=float, nargs="+", required=True,
                      metavar="GEV", help="Dark pion mass values [GeV]")

    ctau_kappa = grid.add_mutually_exclusive_group(required=True)
    ctau_kappa.add_argument("--ctau", type=float, nargs="+",
                             metavar="MM", help="Target c·τ values [mm]")
    ctau_kappa.add_argument("--kappa", type=float, nargs="+",
                             help="Coupling κ values")

    grid.add_argument("--fD", type=float, default=None,
                      help="Dark decay constant [GeV] (default: mPiD)")
    grid.add_argument("--r-piD", type=float, default=0.6, dest="r_piD",
                      help="m_piD / Λ_D ratio (default: 0.6).  "
                           "Sets Λ_D, m_qD, and m_ρD via chiral-scaling relations.")
    grid.add_argument("--mDarkQ", type=float, default=None,
                      help="Dark quark mass [GeV] (overrides chiral-scaling default)")
    grid.add_argument("--mRhoD", type=float, default=None,
                      help="Dark rho mass [GeV] (overrides chiral-scaling default)")
    grid.add_argument("--species", default="T15",
                      choices=["T15", "T8", "T3", "(0,1)", "(0,2)", "(1,2)"],
                      help="Reference pion species for c·τ target (default: T15)")
    grid.add_argument("--nevents", type=int, default=10000,
                      help="Events per job (default: 10000)")
    grid.add_argument("--sqrts", type=float, default=13600.0,
                      help="Centre-of-mass energy [GeV] (default: 13600)")

    # --- Paths ---
    paths = p.add_argument_group("paths")
    paths.add_argument("--process-dir", required=True,
                       help="Compiled MadGraph process directory")
    paths.add_argument("--runs-dir", default="/ourdisk/hpc/ouhep/jburzyns/dont_archive/model_benchmarking_runs",
                       help="Directory for per-point isolated run dirs")
    paths.add_argument("--cards-base", default="cards",
                       help="Base directory where write_cards.py writes cards (default: cards/)")
    paths.add_argument("--logs-dir", default="logs",
                       help="Directory for Slurm stdout/stderr logs (default: logs/)")
    paths.add_argument("--mg5-exe", default="mg5_aMC",
                       help="MadGraph executable (default: mg5_aMC)")

    # --- Environment ---
    env = p.add_argument_group("environment")
    env.add_argument("--env-setup", default="",
                     help="Command(s) to set up the environment in each job "
                          "(e.g. 'source /cvmfs/.../setup.sh'). "
                          "Run before MadGraph is called.")

    # --- Slurm options (none hardcoded) ---
    slurm = p.add_argument_group("slurm resource options")
    slurm.add_argument("--partition", default="sooner_test",
                       help="Slurm partition / queue name (default: sooner_test)")
    slurm.add_argument("--account", default=None,
                       help="Slurm account / project to charge")
    slurm.add_argument("--time", default=None,
                       help="Wall-clock time limit (e.g. 04:00:00)")
    slurm.add_argument("--mem", default="16G",
                       help="Memory per node (default: 16G)")
    slurm.add_argument("--container", default="el9hw",
                       help="Slurm container image (default: el9hw)")
    slurm.add_argument("--cpus-per-task", default=None,
                       help="CPUs per task (default: cluster default)")
    slurm.add_argument("--constraint", default=None,
                       help="Node feature constraint (e.g. 'el9')")
    slurm.add_argument("--sbatch-opt", action="append", default=[],
                       metavar="'--key=value'",
                       help="Any additional #SBATCH option, e.g. "
                            "--sbatch-opt='--nodes=1'. Can be repeated.")

    # --- Behaviour ---
    p.add_argument("--dry-run", action="store_true",
                   help="Print job scripts and commands without submitting")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Validate process directory
    if not os.path.isdir(args.process_dir):
        print(f"ERROR: process directory not found: {args.process_dir}", file=sys.stderr)
        print("Generate it first with:  mg5_aMC proc_card.txt", file=sys.stderr)
        sys.exit(1)

    # Build grid: Cartesian product of (mXd, mPiD, ctau-or-kappa)
    varying_param = "ctau" if args.ctau else "kappa"
    varying_vals  = args.ctau if args.ctau else args.kappa

    grid = list(itertools.product(args.mXd, args.mPiD, varying_vals))
    print(f"\nParameter grid: {len(grid)} points  "
          f"({len(args.mXd)} mXd × {len(args.mPiD)} mPiD × "
          f"{len(varying_vals)} {varying_param})")

    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.runs_dir, exist_ok=True)

    # Collect Slurm resource opts (only those the user actually set)
    sbatch_opts = {
        "partition":     args.partition,
        "account":       args.account,
        "time":          args.time,
        "mem":           args.mem,
        "container":     args.container,
        "cpus-per-task": args.cpus_per_task,
        "constraint":    args.constraint,
    }

    submitted = 0
    skipped   = 0

    for mXd, mPiD, val in grid:
        if varying_param == "ctau":
            point_name = f"mXd{mXd:.0f}_mPiD{mPiD:.0f}_ctau{val:.0f}"
            wp_args    = ["--ctau", str(val)]
        else:
            point_name = f"mXd{mXd:.0f}_mPiD{mPiD:.0f}_kappa{val:.4g}"
            wp_args    = ["--kappa", str(val)]

        cards_dir = os.path.join(args.cards_base, point_name)

        # --- Step 1: generate cards ---
        write_cmd = [
            sys.executable, "write_cards.py",
            "--mXd",    str(mXd),
            "--mPiD",   str(mPiD),
            "--nevents", str(args.nevents),
            "--sqrts",  str(args.sqrts),
            "--species", args.species,
            "--process-dir", args.process_dir,
            "--mg5-exe", args.mg5_exe,
            "--output", cards_dir,
            *wp_args,
        ]
        if args.fD     is not None: write_cmd += ["--fD",     str(args.fD)]
        write_cmd += ["--r-piD", str(args.r_piD)]
        if args.mDarkQ is not None: write_cmd += ["--mDarkQ", str(args.mDarkQ)]
        if args.mRhoD  is not None: write_cmd += ["--mRhoD",  str(args.mRhoD)]

        if args.dry_run:
            print(f"\n[dry-run] Would generate cards: {' '.join(write_cmd)}")
        else:
            print(f"\nGenerating cards for {point_name} ...", end=" ", flush=True)
            result = subprocess.run(write_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FAILED\n{result.stderr.strip()}")
                skipped += 1
                continue
            print("done")

        # --- Step 2: build and submit Slurm job ---
        script = build_job_script(
            point_name  = point_name,
            cards_dir   = cards_dir,
            process_dir = args.process_dir,
            runs_dir    = args.runs_dir,
            logs_dir    = args.logs_dir,
            mg5_exe     = args.mg5_exe,
            env_setup   = args.env_setup,
            sbatch_opts = sbatch_opts,
            extra_sbatch = args.sbatch_opt,
        )

        if args.dry_run:
            print(f"\n[dry-run] Slurm script for {point_name}:")
            print("\n".join("    " + l for l in script.splitlines()))
        else:
            result = subprocess.run(
                ["sbatch"], input=script, text=True, capture_output=True
            )
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"  Submitted job {job_id} for {point_name}")
                submitted += 1
            else:
                print(f"  sbatch FAILED for {point_name}:\n{result.stderr.strip()}")
                skipped += 1

    if not args.dry_run:
        print(f"\nSubmitted {submitted} jobs, skipped {skipped}.")
        if submitted:
            print(f"Logs:   {os.path.abspath(args.logs_dir)}/")
            print(f"Events: {os.path.abspath(args.runs_dir)}/<point>/Events/<point>/")


if __name__ == "__main__":
    main()
