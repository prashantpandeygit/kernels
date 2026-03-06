from cgi import test
from numpy import block
import torch
from torch._refs import new_empty
import triton
import triton.language as tl

DEVICE = torch.device(torch.cuda.current_device())

@triton.jit # to tell triton to compile function into gpu
# block size param is static and can be known at compile time (does not change) 
def add_kernel(p_ptr, q_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # multiple programs process, every program is unique
    # so each program needs to have a unique set of operations it needs to perform
    pid = tl.program_id(axis=0) #axis 0 as we are working with 1D grid
    # if vec length = 256
    # block size = 64
    # so pid 0 will process 0-63 indices, and so on

    # calculating range for each pid
    block_start = pid * BLOCK_SIZE
    # offset is array of int32 that are pointers
    offset = block_start + tl.arange(0, BLOCK_SIZE) #if pid=1, then offset = [64,65,...126,127]
    # a mask to save memory if we have more vector that need a new block (if not a multiple of block size)
    mask = offset < n_elements
    # load data from memory to SRAM
    p = tl.load(p_ptr + offset, mask = mask)  
    q = tl.load(q_ptr + offset, mask = mask)

    output = p+q

    # save data back to dram
    tl.store(output_ptr + offset, output, mask=mask)

def add (p: torch.Tensor, q: torch.Tensor): #WRAPPER
    output = torch.empty_like(p)

    assert p.device == DEVICE and q.device == DEVICE and output.device == DEVICE

    # getting length of the vectors
    n_elements = output.numel() # return total entries in a tensor of any shape

    # defining grid
    # grid defined no of kernel instance that run in parallel
    # 1 grid here incase of addition of two vectors

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    #triton.cdiv = (n_elements + (BLOCK_SIZE - 1)) // BLOCK_SIZE (we get blocks in a grid)
    # also block size param that is passed at compile time, not runtime

    add_kernel[grid](p, q, output, n_elements, BLOCK_SIZE=1024)

    return output


def test_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE):
    torch.manual_seed(0)
    p = torch.rand(size, device="cuda")
    q = torch.rand(size, device="cuda")

    r_triton = add(p,q)
    r_torch = p + q

    torch.testing.assert_close(r_triton, r_torch, atol=atol, rtol=rtol)
    print('test passed')

if __name__ == "__main__":
    test_kernel(size=4096)
    test_kernel(size=4097)
    test_kernel(size=238297)