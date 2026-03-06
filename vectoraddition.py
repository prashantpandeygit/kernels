import torch
import triton
import triton.language as tl


def test(size, atol, rtol):
    torch.manual_seed(0)
    p = torch.rand(size, device="cuda")
    q = torch.rand(size, device="cuda")

    r_triton = add(p,q)
    r_torch = p + q

    torch.testing.assert_close(r_triton, r_torch, atol, rtol)


def add (p: torch.Tensor, q: torch.Tensor):
    # helper/wrapper function
    output = torch.empty_like(p)
    