import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, m) + tl.load(y_ptr + offs, m), m)

n = 4096
x = torch.randn(n, device='cuda', dtype=torch.float32)
y = torch.randn_like(x)
out = torch.empty_like(x)

grid = (triton.cdiv(n, 128),)
add_kernel[grid](x, y, out, n, BLOCK=128)

torch.testing.assert_close(out, x + y)
print("OK:", torch.cuda.get_device_name(0), "Triton", triton.__version__)
