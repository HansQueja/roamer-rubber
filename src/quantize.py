# src/quantize.py
import os
import copy
import torch
import torch.nn as nn


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """
    Apply PyTorch dynamic quantization to Linear and Conv2d layers.
    Dynamic quantization is the safest option — no calibration data needed,
    works on CPU, and typically reduces model size by 50-75% with minimal
    accuracy loss on classification tasks.

    Returns a new quantized model; original is unchanged.
    """
    model_copy = copy.deepcopy(model).cpu()
    model_copy.eval()

    quantized = torch.quantization.quantize_dynamic(
        model_copy,
        qconfig_spec={nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    return quantized


def get_model_size_mb(model: nn.Module,
                       tmp_path: str = "/tmp/_size_check.pth") -> float:
    """Save model to a temp file and measure size in MB."""
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return round(size_mb, 3)


def get_param_count(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    return {"total": total, "trainable": trainable}