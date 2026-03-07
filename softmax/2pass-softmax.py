import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    dim_size,
    inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx   = tl.program_id(0)
    outer_idx = row_idx // inner_size
    inner_idx = row_idx  % inner_size
    base      = outer_idx * dim_size * inner_size + inner_idx

    m_i = -float("inf")   # running max
    l_i = 0.0             # running sum of exp

    for start in range(0, dim_size, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim_size
        x = tl.load(
            input_ptr + base + cols * inner_size,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        m_new  = tl.maximum(m_i, tl.max(x, axis=0))
        # Rescale old sum + add new block
        l_i    = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m_i    = m_new

# normalize and write
    for start in range(0, dim_size, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < dim_size
        x = tl.load(
            input_ptr + base + cols * inner_size,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        out = tl.exp(x - m_i) / l_i
        tl.store(output_ptr + base + cols * inner_size, out, mask=mask)


# input, output, shape are device tensors
def solution(input, dim: int, output, shape, ndim: int):
    shape_list = shape.tolist()
    dim_size   = int(shape_list[dim])

    # inner_size  = product of dims after dim
    inner_size = 1
    for i in range(dim + 1, ndim):
        inner_size *= int(shape_list[i])

    # outer_size  = product of dims befire dim
    outer_size = 1
    for i in range(dim):
        outer_size *= int(shape_list[i])

    n_rows     = outer_size * inner_size
    # Cap at 4096 so register pressure stays manageable;
    BLOCK_SIZE = min(triton.next_power_of_2(dim_size), 4096)

    softmax_kernel[(n_rows,)](
        input, output,
        dim_size, inner_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4 if BLOCK_SIZE <= 512 else 8,
    )