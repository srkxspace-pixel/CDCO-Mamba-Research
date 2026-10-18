"""Reliability-aware adaptive fusion (RAF) and dynamic per-modality feature dimension (DFD).

Reliability: per patient, an evidential head on each modality's aligned tokens outputs Dirichlet
evidence over the classes (Han et al., Trusted Multi-View Classification, ICLR 2021). Its
uncertainty u = K / sum(alpha) gives reliability r = 1 - u, i.e. how decisive that modality is on
its own. A missing modality has r = 0. Reliability is detached: only the evidential loss trains
the heads, so r keeps that meaning instead of becoming a free gate.

RAF: relative weight a_m = r_m / max(r_sMRI, r_fMRI). The more reliable modality passes unchanged
(a = 1); the other is blended toward a learned stand-in token, H_m <- a_m H_m + (1 - a_m) t_m.

DFD: each modality keeps its k_m = k_min + (d - k_min) a_m leading feature dims (fractional
boundary dim). Nested dropout during training (random extra truncation) orders the dims by
importance, so a less reliable modality is compressed to fewer dims rather than passed at
full width.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def edl_loss(evidence, y, kl_weight):
    """Expected cross-entropy under Dir(alpha) plus annealed KL(Dir(alpha~) || Dir(1)) on the
    non-target evidence (TMC). y may be soft (mixup)."""
    alpha = evidence + 1
    S = alpha.sum(-1, keepdim=True)
    ce = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(-1)
    a = y + (1 - y) * alpha
    Sa = a.sum(-1)
    k = torch.tensor(float(a.shape[-1]), device=a.device)
    kl = (torch.lgamma(Sa) - torch.lgamma(k) - torch.lgamma(a).sum(-1)
          + ((a - 1) * (torch.digamma(a) - torch.digamma(Sa)[:, None])).sum(-1))
    return (ce + kl_weight * kl).mean()


class ReliabilityGate(nn.Module):
    """Gates the aligned token sets [sMRI (b, c, d), fMRI (b, c, d)] by per-patient reliability."""

    def __init__(self, d, n_cls, raf=True, dfd=True, min_dim=12, nested_dropout=0.3):
        super().__init__()
        self.raf, self.dfd, self.min_dim, self.nested_dropout = raf, dfd, min_dim, nested_dropout
        self.heads = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_cls)) for _ in range(2))
        # learned stand-ins for sMRI, fMRI; not zero: an all-zero token set zeroes the Cross-Mamba gate
        # (silu(in_proj(0)) = 0) and the contrastive F.normalize then has a 1/eps (1e12) gradient
        self.missing = nn.Parameter(torch.randn(2, 1, 1, d))
        self.info = {}

    def width(self, a, d):
        k = self.min_dim + (d - self.min_dim) * a
        if self.training and self.nested_dropout > 0:
            cut = torch.rand_like(k) < self.nested_dropout
            k = torch.where(cut, torch.minimum(k, self.min_dim + (d - self.min_dim) * torch.rand_like(k)), k)
        return k

    def forward(self, tokens, avail):
        evidence = [F.softplus(h(t.mean(1))) for h, t in zip(self.heads, tokens)]
        n_cls = evidence[0].shape[-1]
        r = torch.stack([1 - n_cls / (e + 1).sum(-1) for e in evidence], 1).detach() * avail
        a = r / r.max(1, keepdim=True).values.clamp_min(1e-6)
        out, dims = [], []
        for m, t in enumerate(tokens):
            am, d = a[:, m, None, None], t.shape[-1]
            if self.raf:
                t = am * t + (1 - am) * self.missing[m]
            else:  # missing modality still replaced, but no reliability weighting
                t = torch.where(avail[:, m, None, None], t, self.missing[m].expand_as(t))
            k = self.width(a[:, m], d) if self.dfd else torch.full_like(a[:, m], d)
            t = t * torch.clamp(k[:, None] - torch.arange(d, device=t.device), 0, 1)[:, None, :]
            out.append(t)
            dims.append(k)
        self.info = dict(evidence=evidence, avail=avail, reliability=r, weight=a, dims=torch.stack(dims, 1))
        return out

    def loss(self, y, kl_weight):
        ev, avail = self.info['evidence'], self.info['avail']
        return sum(edl_loss(e[avail[:, m]], y[avail[:, m]], kl_weight)
                   for m, e in enumerate(ev) if avail[:, m].any())


if __name__ == '__main__':
    torch.manual_seed(0)
    y = torch.tensor([[1., 0.], [0., 1.]])
    right, wrong = torch.tensor([[20., 0.], [0., 20.]]), torch.tensor([[0., 20.], [20., 0.]])
    assert edl_loss(right, y, 1) < edl_loss(torch.zeros(2, 2), y, 1) < edl_loss(wrong, y, 1)

    g = ReliabilityGate(d=48, n_cls=2, min_dim=12).eval()
    s, f = torch.randn(3, 5, 48), torch.randn(3, 5, 48)
    avail = torch.tensor([[True, True], [True, False], [False, True]])
    gs, gf = g([s, f], avail)
    w, k = g.info['weight'], g.info['dims']
    assert torch.allclose(w.max(1).values, torch.ones(3)), w          # most reliable modality kept whole
    assert w[1, 1] == 0 and w[2, 0] == 0                               # missing -> weight 0
    assert torch.allclose(gf[1], g.missing[1, 0].expand(5, 48) * (torch.arange(48) < 12))  # stand-in, 12 dims
    assert k.min() >= 12 and k.max() <= 48 and (k[:, 0] >= k[:, 1]).tolist() == (w[:, 0] >= w[:, 1]).tolist()
    assert (gs[:, :, k[:, 0].floor().long().max() + 1:] == 0).all()   # dims beyond k are cut
    print('reliability gate checks passed; weights', w.tolist(), 'dims', k.tolist())
