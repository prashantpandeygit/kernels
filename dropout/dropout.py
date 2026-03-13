import torch
import triton
import triton.language as tl

DEVICE = torch.device('cuda:0')

def wrapper(x,p,seed):
    output=torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid=lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    kernel[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output #not returning mask as no bwd pass for now

@triton.jit
def kernel(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE:tl.constexpr,):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr+offset, mask=mask, other = 0.0)
    #uniform dist array of [0,1] for p (randn in case of torch)
    random = tl.rand(seed, offset)
    #prune
    x_keep = random >= p #true or false
    scale = 1/(1-p)
    output = x * x_keep * scale#where xkeep is true, the value is the bern. formula, else 0
    #back to DRAM 
    tl.store(output_ptr+offset, output, mask=mask)


x = torch.randn(size = (6, ), device=DEVICE)
test_output = wrapper(x, p=0.5, seed = 111)
print(x, test_output)

