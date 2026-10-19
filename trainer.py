import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    use_amp=True,
    amp_dtype=torch.bfloat16
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    total_grad_norm = 0.0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.time()

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()

        # track unscaled gradient norm for numerical stability diagnostics
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf")).item()
        total_grad_norm += grad_norm

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_time = time.time() - start_time
    samples_per_sec = total / elapsed_time if elapsed_time > 0 else 0.0

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    avg_grad_norm = total_grad_norm / len(loader)

    return avg_loss, acc, avg_grad_norm, elapsed_time, samples_per_sec


def evaluate(
    model,
    loader,
    criterion,
    device,
    use_amp=True,
    amp_dtype=torch.bfloat16
):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def plot_training_curves(history, title_prefix=""):
    epochs_range = range(1, len(history["train_loss"]) + 1)
    prefix = f"{title_prefix} " if title_prefix else ""

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    plt.plot(epochs_range, history["val_loss"], label="Val Loss", color="crimson", lw=2, linestyle="--")
    plt.title(f"{prefix}Loss vs. Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], label="Train Acc", color="royalblue", lw=2)
    plt.plot(epochs_range, history["val_acc"], label="Val Acc", color="green", lw=2, linestyle="--")
    plt.title(f"{prefix}Top-1 Accuracy vs. Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.show()


def train(
    model,
    train_loader,
    val_loader,
    optimizer=None,
    criterion=None,
    scheduler=None,
    epochs=30,
    lr=0.01,
    device="cuda",
    use_amp=True,
    amp_dtype=torch.bfloat16,
    seed=0,
    model_name=None,
    plot_curves=True
):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    name = model_name or model.__class__.__name__
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # default hyperparameters matching table n.4
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "grad_norm": [],
        "epoch_time": []
    }

    gpu_label = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Starting Training ({name}) on {gpu_label}...")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, grad_norm, elapsed, samples_per_sec = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype
        )

        scheduler.step()

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["grad_norm"].append(grad_norm)
        history["epoch_time"].append(elapsed)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"({elapsed:.1f}s, {samples_per_sec:.0f} samples/s) | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.2f}% | "
            f"Grad Norm: {grad_norm:.2f}"
        )

    print("\nTraining Finished!")

    if plot_curves:
        plot_training_curves(history, title_prefix=name)

    return history
