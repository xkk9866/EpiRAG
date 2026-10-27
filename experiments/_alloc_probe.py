import torch

for index in (0, 1):
    free, total = torch.cuda.mem_get_info(index)
    print(f"cuda:{index} free_mib {free / 1024**2:.0f} total_mib {total / 1024**2:.0f}", flush=True)
    small = torch.empty((256, 1024, 1024), dtype=torch.float16, device=f"cuda:{index}")
    print(f"cuda:{index} small_ok", flush=True)
    big = torch.empty((7 * 1024, 1024, 1024), dtype=torch.float16, device=f"cuda:{index}")
    print(f"cuda:{index} big_ok {big.numel() * 2 / 1024**3:.1f} GiB", flush=True)
    del small, big
    torch.cuda.empty_cache()
print("alloc_done", flush=True)
