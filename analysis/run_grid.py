#!/usr/bin/env python
"""Run extract.py over (a subset of) the grid, locally in parallel or as a Slurm array.

    # local, 4 parallel workers, only points not yet done
    python run_grid.py --mode local -j 4 --samples onshell --mX 2000 --mPiD 12
    # whole grid as one Slurm array job (one task per file)
    python run_grid.py --mode slurm
    python run_grid.py --mode slurm --dry-run     # just write the script, print it

Each extraction streams events (memory ~0.5-1 GB) and takes ~1 min per
1000-event file. Logs go to $EJ_OUT/logs/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common as C

HERE = Path(__file__).resolve().parent


def todo(points, force):
    return [p for p in points if force or not (p.outdir / "meta.json").exists()]


def run_local(points, jobs, py, force):
    logdir = C.OUT / "logs"
    logdir.mkdir(parents=True, exist_ok=True)

    def one(p):
        log = logdir / f"{p.sample}__{p.name}.log"
        cmd = [py, str(HERE / "extract.py"), "--sample", p.sample, "--point", p.name]
        if force:
            cmd.append("--force")
        t = time.time()
        with open(log, "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
        print(f"[{'ok' if rc == 0 else f'FAIL rc={rc}'}] {p.sample}/{p.name} "
              f"({time.time() - t:.0f} s)  log: {log}", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        rcs = list(ex.map(one, points))
    return int(any(rcs))


def run_slurm(points, a):
    logdir = C.OUT / "logs" / "slurm"
    logdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tasks = C.OUT / "logs" / f"tasks_{stamp}.txt"
    tasks.write_text("".join(f"{p.sample} {p.name}\n" for p in points))
    force = " --force" if a.force else ""
    site = "".join(f"#SBATCH --{k}={v}\n" for k, v in
                   (("partition", a.partition), ("account", a.account), ("container", a.container)) if v)
    site += "".join(f"#SBATCH {o}\n" for o in a.sbatch_opt)
    script = f"""#!/usr/bin/env bash
#SBATCH --job-name=ej_ana
{site}#SBATCH --time={a.time}
#SBATCH --mem={a.mem}
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --array=1-{len(points)}%{a.max_parallel}
#SBATCH --output={logdir}/%A_%a.out
#SBATCH --error={logdir}/%A_%a.err
source {HERE}/env.sh
read SAMPLE POINT <<< "$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {tasks})"
echo "task $SLURM_ARRAY_TASK_ID: $SAMPLE $POINT on $(hostname)"
cd {HERE}
$EJ_PY extract.py --sample "$SAMPLE" --point "$POINT"{force}
"""
    sfile = C.OUT / "logs" / f"submit_{stamp}.sh"
    sfile.write_text(script)
    print(f"{len(points)} tasks -> {tasks}\nscript: {sfile}")
    if a.dry_run:
        print(script)
        return 0
    return subprocess.call(["sbatch", str(sfile)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["local", "slurm"], default="local")
    ap.add_argument("--samples", nargs="+", default=["onshell", "merged"])
    ap.add_argument("--mX", nargs="+", type=int)
    ap.add_argument("--mPiD", nargs="+", type=int)
    ap.add_argument("--ctau", nargs="+", type=int)
    ap.add_argument("--points", nargs="+", help="explicit point names (applied after the filters)")
    ap.add_argument("--force", action="store_true", help="redo points that already have meta.json")
    ap.add_argument("-j", "--jobs", type=int, default=4, help="local parallel workers")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--partition", default=None, help="Slurm partition (default: cluster default)")
    ap.add_argument("--account", default=None, help="Slurm account")
    ap.add_argument("--container", default=None, help="Slurm container image, if your site uses one")
    ap.add_argument("--sbatch-opt", action="append", default=[],
                    help="extra #SBATCH directive, e.g. --sbatch-opt='--qos=short' (repeatable)")
    ap.add_argument("--time", default="01:00:00")
    ap.add_argument("--mem", default="4G")
    ap.add_argument("--max-parallel", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    samples = [C.SHORT_TO_SAMPLE.get(s, s) for s in a.samples]
    pts = C.discover(samples=samples, mX=a.mX, mPiD=a.mPiD, ctau=a.ctau)
    if a.points:
        pts = [p for p in pts if p.name in a.points]
    pts = todo(pts, a.force)
    if not pts:
        print("nothing to do")
        return 0
    print(f"{len(pts)} files to process")
    if a.mode == "local":
        if a.dry_run:
            for p in pts:
                print(p.sample, p.name)
            return 0
        return run_local(pts, a.jobs, a.python, a.force)
    return run_slurm(pts, a)


if __name__ == "__main__":
    sys.exit(main())
