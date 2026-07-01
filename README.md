# poker44-skip

Poker44 Bittensor subnet (netuid 126) miner — fork of the top-performing bump miner architecture combined with multi-miner fleet management.

## What's different from the upstream bump miner

| | Upstream (`pocker44-miner`) | This repo |
|---|---|---|
| Manifest identity | Derived from live `git HEAD` at startup | **Hardcoded per-miner** via env vars (`POKER44_MANIFEST_REPO_URL`, `POKER44_MANIFEST_REPO_COMMIT`) — stable, predictable, no git dependency |
| Multi-miner config | 4 separate `ecosystem.*.config.js` files | **Single `run_miners.sh`** with an `add_miner` registry — one file to rule them all |
| Scanner noise | Not filtered | Suppressed (`_ScannerNoiseFilter`) to reduce log spam |

Everything else (inference, feature engineering, topk head, BumpModel, training scripts) is unchanged from the top miner.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Install the Poker44 subnet package (provides BaseMinerNeuron, DetectionSynapse, etc.)
git clone https://github.com/Poker44/Poker44-subnet && pip install -e Poker44-subnet
```

## Train a model

```bash
# Place benchmark data in data/ (e.g. data/training_benchmark_*.txt or benchmark JSONs)
python train_bump.py          # produces models/bump_model.joblib
python train_v10.py           # v10 averaged ensemble  → models/bump_model_v10.joblib
python train_v19.py           # v19 sanitization-invariant → models/bump_model_v19.joblib
```

## Configure the fleet

Edit **`run_miners.sh`** — the `MINER REGISTRY` section:

1. Update `ALLOWED_VALIDATOR_HOTKEYS` if the validator set changes.
2. For each miner, call `add_miner` with:
   - PM2 name, wallet coldkey, hotkey, axon port
   - Model file (basename under `models/`)
   - **Manifest identity** — `repo_url`, `repo_commit`, `model_name`, `model_version`
   - Optional per-miner env overrides (heredoc block), e.g. `POKER44_TOPK_FRAC=0.15`

```bash
add_miner \
  "sk-1" \               # pm2 name
  "skip" \               # wallet coldkey
  "sk1" \                # hotkey
  "11106" \              # axon port
  "bump_model_v19.joblib" \
  "https://github.com/yourname/poker44-skip" \   # manifest repo_url (hardcoded)
  "abc1234" \            # manifest repo_commit  (hardcoded)
  "poker44-bump-robust" \
  "bump-saninv-v19" \
  <<'MINER_ENV'
POKER44_TOPK_FRAC=0.15
MINER_ENV
```

### Manifest identity fields

| Env var | Purpose |
|---|---|
| `POKER44_MANIFEST_REPO_URL` | Broadcast as `repo_url` in the manifest |
| `POKER44_MANIFEST_REPO_COMMIT` | Broadcast as `repo_commit` in the manifest |
| `POKER44_MANIFEST_MODEL_NAME` | Model name string |
| `POKER44_MANIFEST_MODEL_VERSION` | Model version string |
| `POKER44_MANIFEST_FRAMEWORK` | Framework string (optional) |

**Update `repo_commit` to match the actual commit you push**, so the validator can verify the code.

## Run the fleet

```bash
./run_miners.sh start      # start all configured miners
./run_miners.sh stop       # stop all miners
./run_miners.sh restart    # stop then start
./run_miners.sh status     # pm2 ls
./run_miners.sh logs sk-1  # tail logs for miner sk-1
```

## Single miner (manual / debug)

```bash
POKER44_BUMP_MODEL=$(pwd)/models/bump_model_v19.joblib \
POKER44_MANIFEST_REPO_URL=https://github.com/yourname/poker44-skip \
POKER44_MANIFEST_REPO_COMMIT=abc1234 \
POKER44_TOPK_FRAC=0.15 \
.venv/bin/python neurons/miner.py \
  --netuid 126 \
  --wallet.name skip --wallet.hotkey sk1 \
  --subtensor.network finney \
  --axon.port 11106 \
  --blacklist.force_validator_permit \
  --logging.info
```

## Key env vars

| Env var | Default | Notes |
|---|---|---|
| `POKER44_BUMP_MODEL` | `models/bump_model.joblib` | Path to the loaded artifact |
| `POKER44_TOPK_FRAC` | `0.15` | Topk head positive fraction (% of batch flagged as bot) |
| `POKER44_CAPTURE` | `0` | Save incoming validator chunks to `live_capture/` |
| `POKER44_MANIFEST_REPO_URL` | compiled-in default | Override per-miner in run_miners.sh |
| `POKER44_MANIFEST_REPO_COMMIT` | compiled-in default | Override per-miner in run_miners.sh |
| `POKER44_SUPPRESS_SCANNER_ERRORS` | `1` | Filter probe/scanner log spam |

## Scoring head

The miner uses the **topk head** (batch-relative safety budget):
- Top `POKER44_TOPK_FRAC` fraction of chunks in a query land in `[0.501, 0.509]` → predicted bot
- Remaining chunks are pushed below `0.49` → predicted human
- Keeps FPR well below the validator's 10% cliff without needing global threshold tuning

## Credits

- Core inference & training pipeline: [phasberg/pocker44-miner](https://github.com/eureka0928/pocker44-miner)
- Vendored baseline features & stacking: [Travis861/Poker44_v1](https://github.com/Travis861/Poker44_v1)
- Multi-miner script pattern: adapted from poker_demo_2

See `VENDOR_NOTICE.md` for full attribution.
