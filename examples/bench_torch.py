import torch, torch.nn as nn, time, sys

def build(d, h, out):
    return nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, out))

def bench(d, h, out, batch, steps, label):
    torch.set_num_threads(1)  # match a single-threaded Rust run for a fair per-step comparison
    model = build(d, h, out)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    lossf = nn.MSELoss()
    x = torch.randn(batch, d)
    y = torch.randn(batch, out)
    # warmup
    for _ in range(5):
        opt.zero_grad()
        l = lossf(model(x), y); l.backward(); opt.step()
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        l = lossf(model(x), y); l.backward(); opt.step()
    dt = time.perf_counter() - t0
    print(f"{label}: d={d} h={h} out={out} batch={batch} steps={steps}  total={dt*1e3:.2f}ms  per_step={dt/steps*1e3:.3f}ms  threads=1")

if __name__ == "__main__":
    bench(16, 32, 4, 8, 2000, "pytorch_tiny")
    bench(64, 128, 10, 32, 1000, "pytorch_small")
    bench(128, 256, 10, 32, 500, "pytorch_medium")
    # also multithreaded for the medium one
    torch.set_num_threads(torch.get_num_threads())  # restore default (all cores)
    print("--- pytorch with all cores (medium) ---")
    bench(128, 256, 10, 32, 500, "pytorch_medium_mt")
