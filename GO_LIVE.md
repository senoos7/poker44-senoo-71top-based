# Poker44 Bump Miner — Go-Live Runbook (SN126)

Status: **published + miner registered; NOT yet serving the axon.**
- Repo: https://github.com/eureka0928/pocker44-miner  (commit `3ef9e55`)
- Miner key: wallet `pes` / hotkey `pes01` → **already registered as uid 102 on netuid 126**
  (ss58 `5D1rspdD7A8HZijXimjE5zNP5bmvuRQYdt1CmxH7FZtWRKCW`; axon currently `0.0.0.0:0`, not served).
- The only remaining step for a live read is **running the miner so it serves the axon.**

## What it scores (verified offline)
- OOF AP 0.86 (≈ leader uid136 live 0.887); FPR 0.000, **0 cliffs** (auto-buffer self-calibrated).
- Local validator (real scorer, unseen windows): epoch-mean reward ~0.67 → projected composite ~0.60.
- Beats king uid32 (composite 0.583); ~parity with provisional leader uid136 (0.599).
- Size-invariant: live chunks (60–120 hands) are subsampled to the trained size (30) and
  bag-averaged over 5 draws, so features match the trained distribution.

## Prerequisites
1. ~~Wallet~~ — done: `pes/pes01` registered (uid 102). No TAO/registration needed.
2. **A server with the axon port reachable from the internet** (validators must connect in).
   - Default port 8091; ensure firewall/security-group allows inbound TCP.
3. ~~Public repo~~ — done (manifest `repo_url`/`repo_commit` self-derive from this checkout at runtime).

## Run the miner  (serves the axon — outward-facing; run only when you intend to go live)
```bash
cd /root/pocker44-miner
pip install -r requirements.txt
pip install -e /root/Poker44-subnet --no-deps   # provides bittensor + poker44.*
# defaults already target pes/pes01:
AXON_PORT=8091 ./run_miner.sh
# or supervised (ecosystem already set to pes/pes01):
pm2 start ecosystem.config.js && pm2 logs poker44_bump_miner
```
The miner self-derives `repo_url`/`repo_commit` from this checkout, so the manifest it serves
will match the published code + committed artifact automatically.

## Read the live result (the whole point)
- First per-window composite appears after ~1 eval window (~10h; needs ≥40 scored chunks).
- Watch our row on the public board:
  `curl -s https://api.poker44.net/api/v1/competition/leaderboard | jq '.data.rows[] | select(.uid==102)'`
  Key fields: `compositeScore`, `reward`, `rankingQuality`, `humanSafetyPenalty` (want 1.0),
  `latencyMeanSeconds`, `windowCompositeScores`.
- Decision gate: if live reward ≥ ~0.68 (composite ≥ ~0.60) and FPR safe (humanSafetyPenalty=1),
  keep running into the next epoch to contest #1. If below, pull back and iterate.

## Timing
- Current epoch ends **2026-06-27 20:00 UTC**; next epoch starts then (120h winner-take-all).
- **For a measurement read: start anytime** — a live composite lands within ~10–20h.
- **To contest a win: be serving before 2026-06-27 20:00 UTC** so you score all 12 windows.

## Active validator hotkeys (permit, stake≥17k — the scorers)
`--blacklist.force_validator_permit` accepts all of these automatically. To pin an explicit
allowlist instead, pass `--blacklist.allowed_validator_hotkeys` with:
```
5E2LP6EnZ54m3wS8s1yPvD5c3xo71kQroBw7aUVK32TKeZ5u
5FxQcdsCXcNjWowQ63Y2oeMhN3JRQksejV3aHRr4XmtknM2k
5FZD47WhA1UaVicYAr7pGnWb2YQLMD7uViipDYN2r1AJ5ggD
5HWe7T96SrY4vRvaLmSoriUJ2CGvhRc559U1vZ1pNPuyz2VA
5G9hfkx9wGB1CLMT9WXkpHSAiYzjZb5o1Boyq4KAdDhjwrc5
5CsvRJXuR955WojnGMdok1hbhffZyB4N5ocrv82f3p5A2zVp
5HmkWGB5PVzKCNLB4QxWWHFVEHPAbKKxGyoXW7Evs38gs126
```

## Maintenance
- Retrain when new benchmark dates release (the conformal buffer auto-recalibrates):
  `python train_bump.py --feature-set ext --calib-window 10`
  then commit + push the new artifact (the miner picks up the new `models/bump_model.joblib`).
- Training data lives in `./data/` (32 released benchmark dates; gitignored).
