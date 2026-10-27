# Source me:  source analysis/env.sh
#
#   EJ_BASE  directory holding runs/<sample>/<point>/ and cards/<sample>/<point>/
#            (default: current directory)
#   EJ_OUT   where features, plots and logs are written (default: $EJ_BASE/analysis_outputs)
#   EJ_PY    python interpreter with the packages in requirements.txt (default: python3)
#
# Site-specific values can go in analysis/env.local.sh (sourced if present, not tracked).
export EJ_ANA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$EJ_ANA_DIR/env.local.sh" ] && source "$EJ_ANA_DIR/env.local.sh"
export EJ_BASE=${EJ_BASE:-$PWD}
export EJ_OUT=${EJ_OUT:-$EJ_BASE/analysis_outputs}
export EJ_PY=${EJ_PY:-python3}
mkdir -p "$EJ_OUT"
