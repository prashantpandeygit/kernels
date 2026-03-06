import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")

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
    p = tl.load(p_ptr + offset, mask = mask, other = None)  
    q = tl.load(q_ptr + offset, mask = mask, other = None)

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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'], # x axis for the plot
        x_vals=[2**i for i in range(12, 28, 1)], # different values of x_names to benchmark
        x_log = True, # makes x-axis logarithmic
        line_arg='provider',
        line_vals=['triton', 'torch'], 
        line_names=['Triton', 'Torch'], 
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s', 
        plot_name='vector-addition-performance',
        args={},
    )
)

def benchmark(size,provider):
    p = torch.rand(size, device=DEVICE, dtype=torch.float32)
    q = torch.rand(size, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.05, 0.95]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: p + q, quantiles=quantiles)
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(p, q), quantiles=quantiles)
    # turning the raw millisecond measurement into meaninful units
    gbps = lambda ms: 3 * p.numel() * p.element_size() * 1e-9 / (ms * 1e-3)
        # 3 = number of memory operations (2 reads + 1 write)
        # x.element_size() = bytes per element (4 for float32, 2 for float16)
        # 1e-9 converts bytes to GB
        # 1e-3 converts milliseconds to seconds
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    test_kernel(size=238297)

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        benchmark.run(save_path='.', print_data=False)