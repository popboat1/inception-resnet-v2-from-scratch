# InceptionResnetV2 Utilities

Modular PyTorch utility library for Inception-ResNet-v2 architectural benchmarking and ablation studies on Imagenette.

---

## Architecture Variants

The library provides a single, unified `InceptionResnetV2` class supporting both spatial stem downsampling and channel attention ablations:

| Variant | Initialization | Description |
| :--- | :--- | :--- |
| **Baseline** | `InceptionResnetV2(stem_type="standard", use_se=False)` | Standard 2016 paper stem ($2\times 2$ final spatial map, ~1.81 GMACs, 56.45M params) |
| **Adapted Stem** | `InceptionResnetV2(stem_type="adapted", use_se=False)` | Preserves $6\times 6$ final spatial map for 128px inputs (~9.66 GMACs, 56.45M params) |
| **SE Attention** | `InceptionResnetV2(stem_type="standard", use_se=True)` | Adds Squeeze-and-Excitation channel attention (~1.82 GMACs, 65.70M params) |
| **Combined** | `InceptionResnetV2(stem_type="adapted", use_se=True)` | Adapted stem + SE Attention (~9.70 GMACs, 65.70M params) |

Convenience aliases:
```python
from InceptionResnetV2 import InceptionResnetV2, InceptionResnetV2_mod, SEInceptionResnetV2
```

---

## Quickstart in Notebook / Colab

### 1. Benchmark GMACs & GPU Memory (1 line)
```python
from InceptionResnetV2 import InceptionResnetV2, benchmark

model = InceptionResnetV2(stem_type="standard")
stats = benchmark(model, device="cuda")
```

### 2. Run an Ablation Training Experiment (3 lines)
```python
from InceptionResnetV2 import InceptionResnetV2, get_imagenette_loaders, train

# 1. load dataset
train_loader, val_loader = get_imagenette_loaders(batch_size=64, num_workers=4)

# 2. instantiate model variant
model = InceptionResnetV2(stem_type="standard", use_se=True)

# 3. train 30 epochs with bf16, grad norm tracking, and auto-plotting
history = train(
    model, 
    train_loader, 
    val_loader, 
    epochs=30, 
    lr=0.01, 
    device="cuda"
)
```
