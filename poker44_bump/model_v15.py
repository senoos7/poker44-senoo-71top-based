"""v15: Deep-Sets hand-encoder + explicit RELATION head (torch), permutation-invariant
over actions (within hand) and hands (within chunk).

Research (2026-06-27): the marginal-match-proof bot signal = intra-set REPETITION ("a
player's hands repeat each other"). Our prior Set Transformer was weak (AP 0.59) because
it lacked a relational inductive bias. v15 fixes that: encode each hand to a vector, then
aggregate the SET with (a) Deep-Sets multi-pool [mean|max|min|std] AND (b) an EXPLICIT
relation signal = mean / high-sim fraction of pairwise cosine similarities of hand vectors.
Heavily regularized (dropout+weight-decay) + hand-subsample augmentation for n=458.

Self-contained featurizer (no dependency on the vendored ChunkSetTransformer). Picklable
wrapper exposes predict_chunk_scores + topk head, drop-in for the bump miner.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Sequence
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # torch optional at import time
    torch = None
    nn = object  # type: ignore

from poker44_bump.model_v5 import _topk_squeeze

# ---- vocabs (ids; 0 reserved as pad) ----
_ACT = {"fold": 1, "check": 2, "call": 3, "bet": 4, "raise": 5}
_STREET = {"preflop": 1, "flop": 2, "turn": 3, "river": 4}
_BUCKET = {"z": 1, "xs": 2, "s": 3, "m": 4, "l": 5, "xl": 6}
N_ACT, N_STREET, N_BUCKET, N_SEAT = 6, 5, 7, 10
MAX_ACTIONS = 14
MAX_HANDS = 80  # cap hands per chunk for latency (deterministic subsample if exceeded)


def _amt_bucket_id(v: float) -> int:
    if v <= 0: return _BUCKET["z"]
    if v <= 0.5: return _BUCKET["xs"]
    if v <= 1: return _BUCKET["s"]
    if v <= 2: return _BUCKET["m"]
    if v <= 5: return _BUCKET["l"]
    return _BUCKET["xl"]


def featurize_chunk(chunk: List[dict]):
    """-> (acts[H,A,4] int64 ids, amt[H,A] float, amask[H,A], hmask[H]) as numpy."""
    hands = list(chunk or [])
    if len(hands) > MAX_HANDS:
        hands = [hands[i] for i in np.linspace(0, len(hands) - 1, MAX_HANDS).astype(int)]
    H = max(1, len(hands))
    acts = np.zeros((H, MAX_ACTIONS, 4), dtype=np.int64)
    amt = np.zeros((H, MAX_ACTIONS), dtype=np.float32)
    amask = np.zeros((H, MAX_ACTIONS), dtype=np.float32)
    hmask = np.zeros((H,), dtype=np.float32)
    for hi, h in enumerate(hands):
        a_list = (h.get("actions") or [])[:MAX_ACTIONS]
        if a_list:
            hmask[hi] = 1.0
        for ai, a in enumerate(a_list):
            a = a or {}
            acts[hi, ai, 0] = _ACT.get(str(a.get("action_type") or "").lower(), 0)
            acts[hi, ai, 1] = _STREET.get(str(a.get("street") or "").lower(), 0)
            bb = float(a.get("normalized_amount_bb", 0) or 0)
            acts[hi, ai, 2] = _amt_bucket_id(bb)
            acts[hi, ai, 3] = min(int(a.get("actor_seat", 0) or 0), N_SEAT - 1)
            amt[hi, ai] = np.tanh(bb / 5.0)
            amask[hi, ai] = 1.0
    return acts, amt, amask, hmask


class DeepSetsRelation(nn.Module):
    def __init__(self, d_act=24, d_hand=32, d_head=64, p=0.3):
        super().__init__()
        self.e_act = nn.Embedding(N_ACT, 8, padding_idx=0)
        self.e_street = nn.Embedding(N_STREET, 4, padding_idx=0)
        self.e_bucket = nn.Embedding(N_BUCKET, 6, padding_idx=0)
        self.e_seat = nn.Embedding(N_SEAT, 4, padding_idx=0)
        self.act_mlp = nn.Sequential(nn.Linear(8 + 4 + 6 + 4 + 1, d_act), nn.ReLU(), nn.Dropout(p))
        self.hand_mlp = nn.Sequential(nn.Linear(2 * d_act, d_hand), nn.ReLU(), nn.Dropout(p))
        self.head = nn.Sequential(
            nn.Linear(4 * d_hand + 2, d_head), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_head, 1))

    def encode_hands(self, acts, amt, amask):
        # acts [B,H,A,4], amt [B,H,A], amask [B,H,A]
        e = torch.cat([self.e_act(acts[..., 0]), self.e_street(acts[..., 1]),
                       self.e_bucket(acts[..., 2]), self.e_seat(acts[..., 3]),
                       amt.unsqueeze(-1)], dim=-1)
        a = self.act_mlp(e)                                   # [B,H,A,d_act]
        m = amask.unsqueeze(-1)
        cnt = m.sum(2).clamp_min(1.0)
        a_mean = (a * m).sum(2) / cnt                         # [B,H,d_act]
        a_max = (a + (m - 1) * 1e9).max(2).values            # masked max
        a_max = torch.nan_to_num(a_max, neginf=0.0)
        hv = self.hand_mlp(torch.cat([a_mean, a_max], dim=-1))  # [B,H,d_hand]
        return hv

    def forward(self, acts, amt, amask, hmask):
        hv = self.encode_hands(acts, amt, amask)             # [B,H,d_hand]
        hm = hmask.unsqueeze(-1)                              # [B,H,1]
        cnt = hm.sum(1).clamp_min(1.0)                        # [B,1]
        s_mean = (hv * hm).sum(1) / cnt
        s_max = (hv + (hm - 1) * 1e9).max(1).values; s_max = torch.nan_to_num(s_max, neginf=0.0)
        s_min = (hv + (1 - hm) * 1e9).min(1).values; s_min = torch.nan_to_num(s_min, posinf=0.0)
        s_var = ((hv - s_mean.unsqueeze(1)) ** 2 * hm).sum(1) / cnt
        s_std = torch.sqrt(s_var + 1e-6)
        # explicit relation: pairwise cosine sim of hand vectors (the repetition signal)
        hn = hv / (hv.norm(dim=-1, keepdim=True) + 1e-6)
        G = torch.bmm(hn, hn.transpose(1, 2))                # [B,H,H]
        pm = hm * hm.transpose(1, 2)                          # valid-pair mask
        eye = torch.eye(G.shape[1], device=G.device).unsqueeze(0)
        pm = pm * (1 - eye)
        npairs = pm.sum((1, 2)).clamp_min(1.0)
        sim_mean = (G * pm).sum((1, 2)) / npairs
        hi_frac = (((G > 0.9).float()) * pm).sum((1, 2)) / npairs
        feat = torch.cat([s_mean, s_max, s_min, s_std,
                          sim_mean.unsqueeze(-1), hi_frac.unsqueeze(-1)], dim=-1)
        return self.head(feat).squeeze(-1)                   # logits [B]


class V15Model:
    """Picklable inference wrapper: holds a trained DeepSetsRelation + topk head."""
    def __init__(self, state_dict, cfg: Dict[str, Any], topk_cfg=None, metadata=None):
        self.state_dict = state_dict
        self.cfg = dict(cfg)
        self.topk_cfg = dict(topk_cfg or {"positive_fraction": 0.15})
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("model_version", "v15-deepsets-relation")
        self.metadata.setdefault("model_name", "poker44-bump-v15")
        self.metadata.setdefault("framework", "deepsets+relation(torch)+topk")
        self.metadata.setdefault("conformal_threshold", 0.5)
        self.metadata["topk_cfg"] = self.topk_cfg
        self.metadata["scoring_head"] = f"topk_v1 (deepsets, positive_fraction={self.topk_cfg.get('positive_fraction')})"
        self.threshold = 0.5
        self.head_mode = "topk"
        self.subsample = False
        self._net = None

    def _net_eval(self):
        if self._net is None:
            net = DeepSetsRelation(**self.cfg)
            net.load_state_dict(self.state_dict)
            net.eval()
            self._net = net
        return self._net

    def predict_raw(self, chunks: Sequence[List[dict]]) -> np.ndarray:
        chunks = [list(c or []) for c in chunks]
        if not chunks:
            return np.zeros((0,), dtype=np.float64)
        net = self._net_eval()
        feats = [featurize_chunk(c) for c in chunks]
        Hmax = max(f[0].shape[0] for f in feats)
        B = len(feats)
        acts = np.zeros((B, Hmax, MAX_ACTIONS, 4), dtype=np.int64)
        amt = np.zeros((B, Hmax, MAX_ACTIONS), dtype=np.float32)
        amask = np.zeros((B, Hmax, MAX_ACTIONS), dtype=np.float32)
        hmask = np.zeros((B, Hmax), dtype=np.float32)
        for i, (a, m, am, hm) in enumerate(feats):
            h = a.shape[0]
            acts[i, :h] = a; amt[i, :h] = m; amask[i, :h] = am; hmask[i, :h] = hm
        with torch.no_grad():
            logits = net(torch.from_numpy(acts), torch.from_numpy(amt),
                         torch.from_numpy(amask), torch.from_numpy(hmask))
            p = torch.sigmoid(logits).numpy().astype(np.float64)
        return p

    def predict_chunk_scores(self, chunks: Sequence[List[dict]]) -> List[float]:
        raw = self.predict_raw(chunks)
        frac = float(os.getenv("POKER44_TOPK_FRAC", self.topk_cfg.get("positive_fraction", 0.15)))
        return _topk_squeeze(raw, frac,
                             float(self.topk_cfg.get("positive_floor", 0.501)),
                             float(self.topk_cfg.get("positive_ceiling", 0.509)),
                             float(self.topk_cfg.get("negative_ceiling", 0.49)))

    def score_chunk(self, chunk: List[dict]) -> float:
        return self.predict_chunk_scores([chunk])[0]
