"""selective_scan_fn from mamba_ssm when installed, else a pure-PyTorch equivalent.

mamba_ssm ships CUDA kernels that do not build on Windows. The fallback follows
mamba_ssm.ops.selective_scan_interface.selective_scan_ref (B, C of shape (b, n, l))
but precomputes everything vectorisable, so the time loop is one addcmul per step.
"""
import torch
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                          delta_softplus=False, return_last_state=False):
        dtype_in = u.dtype
        u, delta, A, B, C = (x.float() for x in (u, delta, A, B, C))
        if delta_bias is not None:
            delta = delta + delta_bias.float()[..., None]
        if delta_softplus:
            delta = F.softplus(delta)
        decay = torch.exp(delta.unsqueeze(-1) * A[:, None, :])               # (b, d, l, n)
        drive = (delta * u).unsqueeze(-1) * B.transpose(1, 2).unsqueeze(1)    # (b, d, l, n)
        h = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
        hs = []
        # unbind, not drive[:, :, t]: each indexing op's backward writes a full-size zero grad (O(l^2)).
        # ponytail: sequential over l (~1 kernel/step); install mamba_ssm on Linux for speed
        for d_t, a_t in zip(drive.unbind(2), decay.unbind(2)):
            h = torch.addcmul(d_t, a_t, h)
            hs.append(h)
        y = (torch.stack(hs, 2) * C.transpose(1, 2).unsqueeze(1)).sum(-1)     # (b, d, l)
        if D is not None:
            y = y + u * D.float()[:, None]
        if z is not None:
            y = y * F.silu(z.float())
        y = y.to(dtype_in)
        return (y, h) if return_last_state else y


if __name__ == "__main__":
    # Compare with a literal transcription of mamba_ssm's selective_scan_ref.
    torch.manual_seed(0)
    b, d, n, l = 2, 6, 4, 33
    u, delta, z = torch.randn(b, d, l), torch.randn(b, d, l), torch.randn(b, d, l)
    A, B, C = -torch.rand(d, n), torch.randn(b, n, l), torch.randn(b, n, l)
    D, bias = torch.randn(d), torch.randn(d)
    dt = F.softplus(delta + bias[..., None])
    x, ys = u.new_zeros(b, d, n), []
    for i in range(l):
        x = torch.exp(dt[:, :, i, None] * A) * x + dt[:, :, i, None] * B[:, None, :, i] * u[:, :, i, None]
        ys.append((x * C[:, None, :, i]).sum(-1))
    ref = (torch.stack(ys, 2) + u * D[:, None]) * F.silu(z)
    out = selective_scan_fn(u, delta, A, B, C, D, z=z, delta_bias=bias, delta_softplus=True)
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()
    print("selective_scan fallback matches reference")
