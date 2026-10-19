import copy
import time
import torch
import torch.nn as nn
try:
    import thop
except ImportError:
    thop = None



def benchmark(
    model,
    optimizer=None,
    criterion=None,
    device="cuda",
    model_name=None,
    input_size=(3, 128, 128),
    batch_size=64,
    num_classes=10,
    num_batches=50,
    batches_per_epoch=133,
    total_epochs=90,
    use_amp=True,
    amp_dtype=torch.bfloat16
):
    name = model_name or model.__class__.__name__
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if thop is None:
        raise ImportError("thop is required for benchmark(). Please install it via 'pip install thop'.")

    # profile on a detached clone so thop hooks never stick to the training model
    model_profile = copy.deepcopy(model).eval()
    x_single = torch.randn(1, *input_size, device=device)

    with torch.no_grad():
        macs, params = thop.profile(model_profile, inputs=(x_single,), verbose=False)
    del model_profile
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    gmac = macs / 1e9
    params_m = params / 1e6

    # default hyperparameters matching table n.4 baseline
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    dummy_inputs = torch.randn(batch_size, *input_size, device=device)
    dummy_targets = torch.randint(0, num_classes, (batch_size,), device=device)

    model.train()
    start_time = time.time()

    for _ in range(num_batches):
        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(dummy_inputs)
                loss = criterion(out, dummy_targets)
        else:
            out = model(dummy_inputs)
            loss = criterion(out, dummy_targets)

        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_sec = time.time() - start_time

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    peak_memory_gb = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else 0.0
    time_per_epoch_min = (elapsed_sec / num_batches) * batches_per_epoch / 60.0
    total_time_hours = time_per_epoch_min * total_epochs / 60.0

    print("==================================================")
    print(f"  Benchmark Report: {name}")
    print("==================================================")
    print(f"  Hardware:          {gpu_name}")
    print(f"  GMAC:              {gmac:.4f} GMACs")
    print(f"  Parameters:        {params_m:.2f} M")
    print(f"  Peak GPU Memory:   {peak_memory_gb:.2f} GB")
    print(f"  Time per Epoch:    {time_per_epoch_min:.2f} minutes ({time_per_epoch_min * 60:.1f}s)")
    print(f"  Estimated ({total_epochs} Ep): {total_time_hours:.2f} hours")
    print("==================================================\n")

    return {
        "model_name": name,
        "gmac": gmac,
        "params_m": params_m,
        "peak_memory_gb": peak_memory_gb,
        "time_per_epoch_min": time_per_epoch_min,
        "total_time_hours": total_time_hours,
        "gpu_name": gpu_name
    }