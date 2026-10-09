"""Chunk SRNL reference derived from pinned upstream chunk.cu.

All five memories read at CHUNK START. G evolves token by token. At chunk end,
matrix memories right-multiply G; vector memories are row vectors times G.
This is not a token-wise TTT approximation. FP64 is retained for gradcheck.
"""
import torch
from torch.nn import functional as F

def srnl_reference(x, query, states, chunk_size):
    """x/query [..., N, D], matrices [...,D,D], vectors [...,D].

    Returns output and all five final states. Leading dimensions broadcast.
    Last partial chunk uses upstream zero-padding semantics, including state.
    """
    if chunk_size < 1 or x.shape != query.shape or x.shape[-2] == 0:
        raise ValueError("Invalid SRNL shape/chunk")
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, q = x.to(dtype), F.normalize(query.to(dtype), dim=-1, eps=1e-12)
        m, v, k, eta_m, alpha_m = (s.to(dtype) for s in states)
        n, d = x.shape[-2:]
        pad = (-n) % chunk_size
        x, q = F.pad(x, (0, 0, 0, pad)), F.pad(q, (0, 0, 0, pad))
        outputs = []
        for start in range(0, n, chunk_size):
            xc, qc = x[..., start:start+chunk_size, :], q[..., start:start+chunk_size, :]
            keys = xc @ k.transpose(-1, -2)
            keys = keys / (keys.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            values = xc @ v.transpose(-1, -2)
            eta = 0.025 * F.softplus((xc * eta_m.unsqueeze(-2)).sum(-1).clamp(-10, 10))
            alpha = ((xc * alpha_m.unsqueeze(-2)).sum(-1) + 2.197224577).clamp(-10, 10).sigmoid()
            outputs.append(qc @ m.transpose(-1, -2))
            g = torch.eye(d, dtype=dtype, device=x.device).expand(*keys.shape[:-2], d, d)
            for i in range(chunk_size):
                a, b = keys[..., i, :], values[..., i, :]
                diff = a - b
                cap = .999 * (1-alpha[..., i]) / ((diff.square().sum(-1)+1e-12).sqrt()+1e-6)
                eff = cap * (-torch.expm1(-eta[..., i] / cap))
                limit = (2 * .999999 * alpha[..., i] / a.square().sum(-1).clamp_min(1e-20))
                # CUDA uses nextafter(limit, 0). On MPS use a conservative <=2 ULP
                # inward move without a per-token CPU synchronization. Port parity
                # is numerical, not bitwise. The unclamped branch is unchanged.
                if x.device.type == 'mps':
                    rounded = limit.detach() * (1-torch.finfo(dtype).eps)
                else:
                    rounded = torch.nextafter(limit.detach(), torch.zeros_like(limit))
                limit = limit + (rounded-limit).detach()
                eff = torch.minimum(eff, limit)
                r = diff + (g @ a.unsqueeze(-1)).squeeze(-1)
                g = alpha[..., i, None, None] * g - eff[..., None, None] * r.unsqueeze(-1) * a.unsqueeze(-2)
            m, v, k = m @ g, v @ g, k @ g
            eta_m = (eta_m.unsqueeze(-2) @ g).squeeze(-2)
            alpha_m = (alpha_m.unsqueeze(-2) @ g).squeeze(-2)
        return torch.cat(outputs, -2)[..., :n, :], (m, v, k, eta_m, alpha_m)
