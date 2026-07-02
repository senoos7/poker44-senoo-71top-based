#!/usr/bin/env bash
# run_miners.sh — Start / stop / restart multiple Poker44 skip miners via PM2.
#
# This script replaces all individual ecosystem.config.js files.
# Each add_miner block is one miner instance. Edit the registry section
# below to configure your fleet, then run:
#
#   ./run_miners.sh start      # start all miners
#   ./run_miners.sh stop       # stop (delete) all miners
#   ./run_miners.sh restart    # stop then start all miners
#   ./run_miners.sh status     # show pm2 ls
#   ./run_miners.sh logs       # list configured miners
#   ./run_miners.sh logs <name> # tail logs for one miner
#
# IMPORTANT: do not put # comment lines inside add_miner \ continuations;
# bash treats the continued line as a comment and drops all arguments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
MINER_SCRIPT="${REPO_ROOT}/neurons/miner.py"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
NETUID="${NETUID:-126}"
NETWORK="${NETWORK:-finney}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# Validator hotkey allowlist — shared by every miner.
ALLOWED_VALIDATOR_HOTKEYS=(
  5E2LP6EnZ54m3wS8s1yPvD5c3xo71kQroBw7aUVK32TKeZ5u
  5EP9fmtknrTnDhQmLRY9ciFYoM7YZM8rPWvQ9J7yywEsn126
  5FxQcdsCXcNjWowQ63Y2oeMhN3JRQksejV3aHRr4XmtknM2k
  5HWe7T96SrY4vRvaLmSoriUJ2CGvhRc559U1vZ1pNPuyz2VA
  5FZD47WhA1UaVicYAr7pGnWb2YQLMD7uViipDYN2r1AJ5ggD
  5Hftk9jrMGSJtKBPWkkAkU53FUSr2BqHGPCThg7mbob3hEq1
  5CsvRJXuR955WojnGMdok1hbhffZyB4N5ocrv82f3p5A2zVp
  5HmkWGB5PVzKCNLB4QxWWHFVEHPAbKKxGyoXW7Evs38gs126
  5G9hfkx9wGB1CLMT9WXkpHSAiYzjZb5o1Boyq4KAdDhjwrc5
)

# Defaults applied to every miner unless overridden in that miner's ENV block.
# POKER44_CAPTURE=1 enables live query capture to live_capture/*.jsonl
# BT_NO_PARSE_CLI_ARGS=false is required so bittensor actually reads --wallet.name
# etc. from the command line (new bittensor versions disable CLI parsing by default).
DEFAULT_MINER_ENV=(
  POKER44_CAPTURE=1
  BT_NO_PARSE_CLI_ARGS=false
  OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1
)

# Manifest identities broadcast to validators (5 unique repo+commit pairs).
# Same values as poker_demo_2/multiple-miner-run.sh
MANIFEST_PUBLIC_NAME="poker44-sota-bot-detection"
MANIFEST_REPO_V2="https://github.com/senoos7/poker44-senoo-v2"
MANIFEST_COMMIT_V2="74cbad2"
MANIFEST_REPO_SOTA_V3="https://github.com/senoos7/poker44-senoo-sota-v3"
MANIFEST_COMMIT_SOTA_V3="1f534ce"
MANIFEST_REPO_BENCHMARK_V5="https://github.com/senoos7/poker44-senoo-benchmark-v5"
MANIFEST_COMMIT_BENCHMARK_V5="70788bb"
MANIFEST_REPO_SUPERVISED_V6="https://github.com/senoos7/poker44-senoo-supervised-v6"
MANIFEST_COMMIT_SUPERVISED_V6="b7750b6"
MANIFEST_REPO_TUNED_V7="https://github.com/senoos7/poker44-senoo-tuned-v7"
MANIFEST_COMMIT_TUNED_V7="8acd9b0"

# =============================================================================
# INTERNAL STATE — do not edit
# =============================================================================
MINER_PM2_NAMES=()
MINER_WALLETS=()
MINER_HOTKEYS=()
MINER_PORTS=()
MINER_MODELS=()
MINER_MANIFEST_REPO_URLS=()
MINER_MANIFEST_REPO_COMMITS=()
MINER_MANIFEST_MODEL_NAMES=()
MINER_MANIFEST_MODEL_VERSIONS=()
MINER_ENV_BLOCKS=()

# =============================================================================
# add_miner FUNCTION
# =============================================================================
# Positional arguments:
#   [1] pm2_name        PM2 process name (pm2 ls / pm2 logs <name>)
#   [2] wallet          coldkey (--wallet.name)
#   [3] hotkey          hotkey (--wallet.hotkey)
#   [4] port            axon port (--axon.port); must be unique per miner
#   [5] model_file      basename under models/ or repo-relative path
#                         e.g. "bump_model_v19.joblib"
#                         e.g. "models/bump_model_v10.joblib"
#   [6] manifest_repo_url     POKER44_MANIFEST_REPO_URL  (hardcoded in manifest)
#   [7] manifest_repo_commit  POKER44_MANIFEST_REPO_COMMIT (hardcoded in manifest)
#   [8] manifest_model_name   POKER44_MANIFEST_MODEL_NAME
#   [9] manifest_model_version POKER44_MANIFEST_MODEL_VERSION
#   [10] MINER_ENV heredoc   per-miner env overrides (key=value, one per line)
#
# The env heredoc supports any env var the miner understands, e.g.:
#   POKER44_TOPK_FRAC=0.15          # topk head positive fraction (default 0.15)
#   POKER44_CAPTURE=0               # disable live capture for this miner
#   POKER44_MANIFEST_FRAMEWORK=...  # override framework string in manifest
#
add_miner() {
  local pm2_name="$1"
  local wallet="$2"
  local hotkey="$3"
  local port="$4"
  local model_file="$5"
  local manifest_repo_url="$6"
  local manifest_repo_commit="$7"
  local manifest_model_name="$8"
  local manifest_model_version="$9"

  MINER_PM2_NAMES+=("$pm2_name")
  MINER_WALLETS+=("$wallet")
  MINER_HOTKEYS+=("$hotkey")
  MINER_PORTS+=("$port")
  MINER_MODELS+=("$model_file")
  MINER_MANIFEST_REPO_URLS+=("$manifest_repo_url")
  MINER_MANIFEST_REPO_COMMITS+=("$manifest_repo_commit")
  MINER_MANIFEST_MODEL_NAMES+=("$manifest_model_name")
  MINER_MANIFEST_MODEL_VERSIONS+=("$manifest_model_version")

  local env_block=""
  if [[ $# -gt 9 ]]; then
    env_block="${10}"
  else
    env_block="$(cat)"
  fi
  MINER_ENV_BLOCKS+=("$env_block")
}

# =============================================================================
# MINER REGISTRY
# =============================================================================
# Fleet configuration — 6 miners across 2 wallets (poker, poker1).
# Model progression: v10 → v10-4 (frac sweep) → v10-4v2 → v10-4v3
#
#   pkn-1  poker  pkn1  11101  bump_model_v10       frac=0.17  (v10 baseline)
#   pkn-2  poker  pkn2  11102  bump_model_v10-4     frac=0.17  (v10-4, conservative)
#   pkn-3  poker  pkn3  11103  bump_model_v10-4     frac=0.19  (v10-4, moderate)
#   pkn-4  poker1 pkn4  11104  bump_model_v10-4     frac=0.22  (v10-4, liberal)
#   pkn-5  poker1 pkn5  11105  bump_model_v10-4v2   frac=0.13  (feat_v2, zero cliffs)
#   pkn-6  poker1 pkn6  11106  bump_model_v10-4v3   frac=0.13  (feat_v3, best AP+Reward)
#
# manifest_model_version values (arg 9) — shortened from full names for brevity:
#   "v10"      (was: bump-ensemble-v10)
#   "v10-4"    (was: bump-ensemble-v10-4)
#   "v10-4v2"  (was: bump-ensemble-v10-4v2)
#   "v10-4v3"  (was: bump-ensemble-v10-4v3)

# --- Miner 1: pkn-1 — v10 diverse averaged ensemble, frac=0.17 ---
add_miner \
  "pkn-1" \
  "poker" \
  "pkn1" \
  "11101" \
  "bump_model_v10.joblib" \
  "${MANIFEST_REPO_V2}" \
  "${MANIFEST_COMMIT_V2}" \
  "${MANIFEST_PUBLIC_NAME}" \
  "v10" \
  <<'MINER_ENV'
POKER44_TOPK_FRAC=0.17
MINER_ENV

# --- Miner 2: pkn-2 — v10-4 large diverse ensemble, frac=0.17 ---
add_miner \
  "pkn-2" \
  "poker" \
  "pkn2" \
  "11102" \
  "bump_model_v10-4.joblib" \
  "${MANIFEST_REPO_SOTA_V3}" \
  "${MANIFEST_COMMIT_SOTA_V3}" \
  "${MANIFEST_PUBLIC_NAME}" \
  "v10-4" \
  <<'MINER_ENV'
POKER44_TOPK_FRAC=0.17
MINER_ENV

# --- Miner 3: pkn-3 — v10-4 large diverse ensemble, frac=0.19 ---
add_miner \
  "pkn-3" \
  "poker" \
  "pkn3" \
  "11103" \
  "bump_model_v10-4.joblib" \
  "${MANIFEST_REPO_BENCHMARK_V5}" \
  "${MANIFEST_COMMIT_BENCHMARK_V5}" \
  "${MANIFEST_PUBLIC_NAME}" \
  "v10-4" \
  <<'MINER_ENV'
POKER44_TOPK_FRAC=0.19
MINER_ENV

# --- Miner 4: pkn-4 — v10-4 large diverse ensemble, frac=0.22 ---
add_miner \
  "pkn-4" \
  "poker1" \
  "pkn4" \
  "11104" \
  "bump_model_v10-4.joblib" \
  "${MANIFEST_REPO_SUPERVISED_V6}" \
  "${MANIFEST_COMMIT_SUPERVISED_V6}" \
  "${MANIFEST_PUBLIC_NAME}" \
  "v10-4" \
  <<'MINER_ENV'
POKER44_TOPK_FRAC=0.22
MINER_ENV

# --- Miner 5: pkn-5 — v10-4v2 (feat_v2+v10-4), frac=0.13 ---
add_miner \
  "pkn-5" \
  "poker1" \
  "pkn5" \
  "11105" \
  "bump_model_v10-4v2.joblib" \
  "${MANIFEST_REPO_TUNED_V7}" \
  "${MANIFEST_COMMIT_TUNED_V7}" \
  "${MANIFEST_PUBLIC_NAME}" \
  "v10-4v2" \
  <<'MINER_ENV'
POKER44_TOPK_FRAC=0.13
MINER_ENV

# --- Miner 6: pkn-6 — v10-4v3 (feat_v3+v10-4), frac=0.13, best AP+Reward ---
add_miner \
  "pkn-6" \
  "poker1" \
  "pkn6" \
  "11106" \
  "bump_model_v10-4v3.joblib" \
  "${MANIFEST_REPO_TUNED_V7}" \
  "${MANIFEST_COMMIT_TUNED_V7}" \
  "${MANIFEST_PUBLIC_NAME}" \
  "v10-4v3" \
  <<'MINER_ENV'
POKER44_TOPK_FRAC=0.13
MINER_ENV

# Add more miners below by copying an add_miner block and editing the fields.
# Remember: each miner needs a unique pm2_name, hotkey, and port.

# =============================================================================
# RUNNER — no need to edit below this line
# =============================================================================

die() { echo "Error: $*" >&2; exit 1; }

require_pm2()  { command -v pm2 >/dev/null 2>&1 || die "pm2 is not installed"; }
require_script() { [[ -f "$MINER_SCRIPT" ]] || die "Miner script not found: $MINER_SCRIPT"; }
require_python() { [[ -x "$PYTHON_BIN" ]] || die "Python venv not found: $PYTHON_BIN"; }

model_path_for() {
  local mf="$1"
  if [[ "$mf" == */* ]]; then
    echo "${REPO_ROOT}/${mf}"
  else
    echo "${REPO_ROOT}/models/${mf}"
  fi
}

apply_env_line() {
  local line="$1"
  line="${line%%#*}"
  line="$(echo "$line" | xargs)"
  [[ -z "$line" ]] && return 0
  [[ "$line" == *=* ]] || die "Invalid env line (expected KEY=VALUE): $line"
  export "$line"
}

apply_env_block() {
  local block="$1"
  local line
  while IFS= read -r line || [[ -n "$line" ]]; do
    apply_env_line "$line"
  done <<<"$block"
}

start_one_miner() {
  local index="$1"
  local pm2_name="${MINER_PM2_NAMES[$index]}"
  local wallet="${MINER_WALLETS[$index]}"
  local hotkey="${MINER_HOTKEYS[$index]}"
  local port="${MINER_PORTS[$index]}"
  local model_file="${MINER_MODELS[$index]}"
  local manifest_repo_url="${MINER_MANIFEST_REPO_URLS[$index]}"
  local manifest_repo_commit="${MINER_MANIFEST_REPO_COMMITS[$index]}"
  local manifest_model_name="${MINER_MANIFEST_MODEL_NAMES[$index]}"
  local manifest_model_version="${MINER_MANIFEST_MODEL_VERSIONS[$index]}"
  local env_block="${MINER_ENV_BLOCKS[$index]}"
  local model_path
  model_path="$(model_path_for "$model_file")"

  # Resolve Vast.ai NAT external port/IP.
  # VAST_TCP_PORT_<internal>=<external> is injected by Vast.ai for each mapped port.
  local vast_env_key="VAST_TCP_PORT_${port}"
  local external_port="${!vast_env_key:-$port}"
  local external_ip="${PUBLIC_IPADDR:-}"

  (
    export PYTHONPATH="${REPO_ROOT}"

    # Apply global defaults
    for line in "${DEFAULT_MINER_ENV[@]}"; do
      apply_env_line "$line"
    done

    # Apply per-miner manifest identity (hardcoded, not from git)
    export POKER44_BUMP_MODEL="${model_path}"
    export POKER44_MANIFEST_REPO_URL="${manifest_repo_url}"
    export POKER44_MANIFEST_REPO_COMMIT="${manifest_repo_commit}"
    export POKER44_MANIFEST_MODEL_NAME="${manifest_model_name}"
    export POKER44_MANIFEST_MODEL_VERSION="${manifest_model_version}"

    # Apply per-miner env overrides
    apply_env_block "$env_block"

    [[ -f "$POKER44_BUMP_MODEL" ]] || \
      echo "Warning: model not found for ${pm2_name}: ${POKER44_BUMP_MODEL}" >&2

    pm2 delete "$pm2_name" 2>/dev/null || true

    pm2 start "$MINER_SCRIPT" \
      --interpreter "$PYTHON_BIN" \
      --name "$pm2_name" \
      --cwd "$REPO_ROOT" \
      --max-restarts 50 \
      --restart-delay 5000 \
      -- \
      --netuid "$NETUID" \
      --neuron.name "$pm2_name" \
      --wallet.name "$wallet" \
      --wallet.hotkey "$hotkey" \
      --subtensor.network "$NETWORK" \
      --axon.port "$port" \
      ${external_ip:+--axon.external_ip "$external_ip"} \
      ${external_port:+--axon.external_port "$external_port"} \
      --logging."$LOG_LEVEL" \
      --blacklist.force_validator_permit \
      --blacklist.allowed_validator_hotkeys "${ALLOWED_VALIDATOR_HOTKEYS[@]}"
  )

  echo "Started ${pm2_name} | wallet=${wallet}/${hotkey} internal=${port} external=${external_ip}:${external_port} model=${model_path}"
}

stop_one_miner() {
  pm2 delete "$1" 2>/dev/null || true
  echo "Stopped $1"
}

start_all() {
  require_pm2; require_script; require_python
  [[ ${#MINER_PM2_NAMES[@]} -gt 0 ]] || die "No miners configured. Add add_miner blocks above."
  for i in "${!MINER_PM2_NAMES[@]}"; do
    start_one_miner "$i"
  done
  pm2 save
  echo ""
  echo "All ${#MINER_PM2_NAMES[@]} miner(s) started. View: pm2 ls"
}

stop_all() {
  require_pm2
  for name in "${MINER_PM2_NAMES[@]}"; do
    stop_one_miner "$name"
  done
  pm2 save
}

status_all() { require_pm2; pm2 ls; }

logs_one() {
  require_pm2
  local target="${1:-}"
  if [[ -n "$target" ]]; then
    pm2 logs "$target"
    return
  fi
  echo "Configured miners:"
  printf '  - %s\n' "${MINER_PM2_NAMES[@]}"
  echo "Run: $0 logs <pm2-name>"
}

ACTION="${1:-start}"
case "$ACTION" in
  start)   start_all ;;
  stop)    stop_all ;;
  restart) stop_all; start_all ;;
  status)  status_all ;;
  logs)    logs_one "${2:-}" ;;
  *)       die "Unknown action: $ACTION (use start|stop|restart|status|logs)" ;;
esac
