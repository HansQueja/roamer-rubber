import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def run_evaluation(model, test_loader, class_names, device, config):
    """
    Runs full evaluation on the test set.
    Returns all_preds and all_labels as numpy arrays
    so they can be reused by the XAI script.
    """
    checkpoint_path = config['paths']['checkpoint']
    output_dir      = config['paths']['outputs']

    # ── Load best checkpoint ───────────────────────────────────────────────────
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))
    model.eval()
    print(f"Loaded checkpoint: '{checkpoint_path}'")

    # ── Run inference on test set ──────────────────────────────────────────────
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            _, predicted   = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Classification report ──────────────────────────────────────────────────
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names)
    print(report)

    # Save report as text
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved → {report_path}")

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Test Set Confusion Matrix')
    plt.xlabel('Predicted Disease')
    plt.ylabel('Actual Disease')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"Saved → {cm_path}")

    return all_preds, all_labels