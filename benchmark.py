import time
import copy
import torch
import torch.nn as nn
import thop

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
    use_amp=True
):
    name = model_name or model.__class__.__name__
    model = model.to(device)
    
    # gmac profiling
    model_profile = copy.deepcopy(model).eval()
    x_single = torch.randn(1, *input_size, device=device)
    
    with torch.no_grad():
        macs, params = thop.profile(model_profile, inputs=(x_single), verbose=False)
    del model_profile  # free memory immediately
    
    gmac = macs / 1e9
    params_m = params / 1e6
    
    # defaults for optim and criterion
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    # reset peak memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
    # dummy batches
    dummy_inputs = torch.randn(batch_size, *input_size, device=device)
    dummy_targets = torch.randint(0, num_classes, (batch_size,), device=device)
    
    # timed train loop
    model.train()
    start_time = time.time()
    
    print(f"\n--- Inception-ResNet-v2 (Adapted Stem) Complexity ---")
    print(f"GMAC: {macs / 1e9:.4f}")
    print(f"Params: {params / 1e6:.2f} M")
    
    model.