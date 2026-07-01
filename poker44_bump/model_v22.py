"""v22 = true action-ORDER sequence model: per-hand Transformer encoder (positional,
masked self-attention over the action window) -> hand vector; cross-hand set-pool
(mean/max/min/std) + explicit REPETITION relation (pairwise cosine sim mean, high-sim
frac, near-exact-dup frac) -> head. Hypothesis: action-order + repetition structure is
the transferable signal (phasberg ships a SetTransformer; d0 no-seq=0.499 < phasberg
seq=0.572). Reuses v15's sanitization-robust featurizer (bucketed amounts, ids).
Picklable, drop-in (predict_raw + topk head). Benchmark-only / honest.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Sequence
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = object  # type: ignore

from poker44_bump.model_v5 import _topk_squeeze
from poker44_bump.model_v15 import (featurize_chunk, MAX_ACTIONS,
                                    N_ACT, N_STREET, N_BUCKET, N_SEAT)


class SeqTransformer(nn.Module):
    def __init__(self, d=32, nhead=4, nlayers=1, d_head=64, p=0.3):
        super().__init__()
        self.e_act = nn.Embedding(N_ACT, d, padding_idx=0)
        self.e_street = nn.Embedding(N_STREET, 8, padding_idx=0)
        self.e_bucket = nn.Embedding(N_BUCKET, 8, padding_idx=0)
        self.e_seat = nn.Embedding(N_SEAT, 8, padding_idx=0)
        self.pos = nn.Embedding(MAX_ACTIONS + 1, d)
        self.in_proj = nn.Linear(d + 8 + 8 + 8 + 1, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=2 * d,
                                           dropout=p, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Sequential(nn.Linear(4 * d + 3, d_head), nn.ReLU(), nn.Dropout(p),
                                  nn.Linear(d_head, 1))

    def encode_hands(self, acts, amt, amask):
        B, H, A, _ = acts.shape
        x = acts.view(B * H, A, 4); am = amt.view(B * H, A); m = amask.view(B * H, A)
        pos = torch.arange(A, device=acts.device).unsqueeze(0).expand(B * H, A)
        e = torch.cat([self.e_act(x[..., 0]), self.e_street(x[..., 1]),
                       self.e_bucket(x[..., 2]), self.e_seat(x[..., 3]),
                       am.unsqueeze(-1)], dim=-1)
        h = self.in_proj(e) + self.pos(pos)
        keypad = (m < 0.5)                      # True = pad position
        allpad = keypad.all(dim=1)
        keypad = keypad.clone(); keypad[allpad, 0] = False   # avoid all-masked rows -> NaN
        z = self.enc(h, src_key_padding_mask=keypad)         # [B*H,A,d]
        mm = m.unsqueeze(-1); cnt = mm.sum(1).clamp_min(1.0)
        hv = (z * mm).sum(1) / cnt                            # masked mean over actions
        return hv.view(B, H, -1)

    def forward(self, acts, amt, amask, hmask):
        hv = self.encode_hands(acts, amt, amask)             # [B,H,d]
        hm = hmask.unsqueeze(-1); cnt = hm.sum(1).clamp_min(1.0)
        s_mean = (hv * hm).sum(1) / cnt
        s_max = (hv + (hm - 1) * 1e9).max(1).values; s_max = torch.nan_to_num(s_max, neginf=0.0)
        s_min = (hv + (1 - hm) * 1e9).min(1).values; s_min = torch.nan_to_num(s_min, posinf=0.0)
        s_std = torch.sqrt(((hv - s_mean.unsqueeze(1)) ** 2 * hm).sum(1) / cnt + 1e-6)
        hn = hv / (hv.norm(dim=-1, keepdim=True) + 1e-6)
        G = torch.bmm(hn, hn.transpose(1, 2))                # [B,H,H] cosine sims
        pm = hm * hm.transpose(1, 2)
        eye = torch.eye(G.shape[1], device=G.device).unsqueeze(0)
        pm = pm * (1 - eye)
        npairs = pm.sum((1, 2)).clamp_min(1.0)
        sim_mean = (G * pm).sum((1, 2)) / npairs
        hi_frac = (((G > 0.9).float()) * pm).sum((1, 2)) / npairs
        dup_frac = (((G > 0.99).float()) * pm).sum((1, 2)) / npairs    # near-exact repetition
        feat = torch.cat([s_mean, s_max, s_min, s_std,
                          sim_mean.unsqueeze(-1), hi_frac.unsqueeze(-1), dup_frac.unsqueeze(-1)], dim=-1)
        return self.head(feat).squeeze(-1)


class V22Model:
    def __init__(self, state_dict, cfg: Dict[str, Any], topk_cfg=None, metadata=None):
        self.state_dict = state_dict
        self.cfg = dict(cfg)
        self.topk_cfg = dict(topk_cfg or {"positive_fraction": 0.15})
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("model_version", "v22-seq-transformer")
        self.metadata.setdefault("model_name", "poker44-bump-v22")
        self.metadata.setdefault("framework", "per-hand-action-transformer + set-pool + repetition-relation (torch) + topk")
        self.metadata.setdefault("conformal_threshold", 0.5)
        self.metadata.setdefault("data_attestation", "No validator-private data used; released benchmark labels only.")
        self.metadata["topk_cfg"] = self.topk_cfg
        self.metadata["scoring_head"] = f"topk_v1 (seq-transformer, positive_fraction={self.topk_cfg.get('positive_fraction')})"
        self.threshold = 0.5
        self.head_mode = "topk"
        self.subsample = False
        self._net = None

    def _net_eval(self):
        if self._net is None:
            net = SeqTransformer(**self.cfg); net.load_state_dict(self.state_dict); net.eval()
            self._net = net
        return self._net

    def predict_raw(self, chunks: Sequence[List[dict]]) -> np.ndarray:
        chunks = [list(c or []) for c in chunks]
        if not chunks:
            return np.zeros((0,), dtype=np.float64)
        net = self._net_eval()
        feats = [featurize_chunk(c) for c in chunks]
        Hmax = max(f[0].shape[0] for f in feats); B = len(feats)
        acts = np.zeros((B, Hmax, MAX_ACTIONS, 4), dtype=np.int64)
        amt = np.zeros((B, Hmax, MAX_ACTIONS), dtype=np.float32)
        amask = np.zeros((B, Hmax, MAX_ACTIONS), dtype=np.float32)
        hmask = np.zeros((B, Hmax), dtype=np.float32)
        for i, (a, m, am, hm) in enumerate(feats):
            h = a.shape[0]; acts[i, :h] = a; amt[i, :h] = m; amask[i, :h] = am; hmask[i, :h] = hm
        out = []
        with torch.no_grad():
            for s in range(0, B, 128):
                e = slice(s, s + 128)
                logits = net(torch.from_numpy(acts[e]), torch.from_numpy(amt[e]),
                             torch.from_numpy(amask[e]), torch.from_numpy(hmask[e]))
                out.append(torch.sigmoid(logits).numpy().astype(np.float64))
        return np.concatenate(out)

    def predict_chunk_scores(self, chunks: Sequence[List[dict]]) -> List[float]:
        raw = self.predict_raw(chunks)
        frac = float(os.getenv("POKER44_TOPK_FRAC", self.topk_cfg.get("positive_fraction", 0.15)))
        return _topk_squeeze(raw, frac,
                             float(self.topk_cfg.get("positive_floor", 0.501)),
                             float(self.topk_cfg.get("positive_ceiling", 0.509)),
                             float(self.topk_cfg.get("negative_ceiling", 0.49)))

    def score_chunk(self, chunk: List[dict]) -> float:
        return self.predict_chunk_scores([chunk])[0]
