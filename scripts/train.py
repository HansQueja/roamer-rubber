import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def _build_scheduler(optimizer, cfg):
    sched = cfg['scheduler']
    stype = sched['type']

    if stype == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = sched['T_max'],
            eta_min = sched.get('eta_min', 1e-6)
        )
    elif stype == 'StepLR':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size = sched['step_size'],
            gamma     = sched['gamma']
        )
    elif stype == 'ReduceLROnPlateau':
        # Bonus option: adapts to val accuracy directly
        # Requires scheduler.step(val_acc) instead of scheduler.step()
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode     = 'max',               # maximising val accuracy
            patience = sched.get('plateau_patience', 3),
            factor   = sched.get('factor', 0.5),
            min_lr   = sched.get('eta_min', 1e-6)
        )
    else:
        raise ValueError(
            f"Unknown scheduler: '{stype}'. "
            f"Supported: CosineAnnealingLR, StepLR, ReduceLROnPlateau"
        )


def train(model, train_loader, val_loader, config, device,
          class_weights=None):
    cfg   = config['training']
    paths = config['paths']

    criterion = (nn.CrossEntropyLoss(weight=class_weights.to(device))
                 if class_weights is not None
                 else nn.CrossEntropyLoss())

    weight_decay_value = config['training'].get('weight_decay', 0.0)

    optimizer = optim.Adam(
        model.parameters(), 
        lr=cfg['learning_rate'], 
        weight_decay=weight_decay_value
    )
    scheduler = _build_scheduler(optimizer, cfg)
    is_plateau = cfg['scheduler']['type'] == 'ReduceLROnPlateau'

    best_val_acc      = 0.0
    epochs_no_improve = 0
    train_loss_hist   = []
    val_acc_hist      = []

    print(f"Training {config['model']['name']} on {device}...")
    print(f"Scheduler  : {cfg['scheduler']['type']}")
    print(f"Early stopping patience: {cfg['patience']} epochs\n")

    for epoch in range(1, cfg['epochs'] + 1):

        # ── Train ──────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # ── Validate ───────────────────────────────────────────
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(model(images), 1)
                val_total   += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        val_acc    = 100 * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        train_loss_hist.append(epoch_loss)
        val_acc_hist.append(val_acc)

        # ReduceLROnPlateau needs the metric; others just step
        if is_plateau:
            scheduler.step(val_acc)
        else:
            scheduler.step()

        # ── Checkpoint & early stopping ────────────────────────
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), paths['checkpoint'])
            marker = f"  ← best ({best_val_acc:.2f}%)"
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg['patience']:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

        print(f"Epoch [{epoch:02d}/{cfg['epochs']}] | "
              f"Loss: {epoch_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}{marker}")

    print(f"\nDone. Best val acc: {best_val_acc:.2f}%")
    _plot_curves(train_loss_hist, val_acc_hist,
                 best_val_acc, config['paths']['outputs'])
    return best_val_acc


def _plot_curves(loss_hist, acc_hist, best_acc, output_dir):
    epochs = range(1, len(loss_hist) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.plot(epochs, loss_hist, color='steelblue')
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.plot(epochs, acc_hist, color='orange')
    ax2.axhline(best_acc, color='red', linestyle='--',
                label=f"Best: {best_acc:.2f}%")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.suptitle("Training Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png",
                dpi=120, bbox_inches='tight')
    plt.show()


def _build_scheduler(optimizer, cfg):
    sched = cfg['scheduler']
    stype = sched['type']

    if stype == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = sched['T_max'],
            eta_min = sched.get('eta_min', 1e-6)
        )
    elif stype == 'StepLR':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size = sched['step_size'],
            gamma     = sched['gamma']
        )
    elif stype == 'ReduceLROnPlateau':
        # Bonus option: adapts to val accuracy directly
        # Requires scheduler.step(val_acc) instead of scheduler.step()
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode     = 'max',               # maximising val accuracy
            patience = sched.get('plateau_patience', 3),
            factor   = sched.get('factor', 0.5),
            min_lr   = sched.get('eta_min', 1e-6)
        )
    else:
        raise ValueError(
            f"Unknown scheduler: '{stype}'. "
            f"Supported: CosineAnnealingLR, StepLR, ReduceLROnPlateau"
        )