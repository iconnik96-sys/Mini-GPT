"""
QLoRA (Quantized Low-Rank Adaptation) Architecture and Hardware Investigation.

This module details:
1. The mathematical and conceptual formulation of QLoRA vs standard LoRA
2. The 4-bit NormalFloat (NF4) data type and Double Quantization
3. Why LoRA adapters can still be trained while base weights are 4-bit quantized
4. Environment probe for bitsandbytes and CUDA hardware compatibility
5. Reproducible QLoRA configuration dictionary for CUDA-enabled environments
"""

from typing import Dict, Any
import torch


def check_qlora_support() -> Dict[str, Any]:
    """
    Probe the local environment to determine whether practical QLoRA 4-bit fine-tuning
    can execute on the current system.
    """
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (No CUDA device)"
    
    bnb_installed = False
    bnb_version = None
    bnb_error = None
    try:
        import bitsandbytes as bnb
        bnb_installed = True
        bnb_version = getattr(bnb, "__version__", "unknown")
    except Exception as e:
        bnb_installed = False
        bnb_error = f"{type(e).__name__}: {str(e)}"
        
    supported = cuda_available and bnb_installed
    
    reasons = []
    if not cuda_available:
        reasons.append("PyTorch is running in CPU-only mode (torch.cuda.is_available() is False).")
    if not bnb_installed:
        reasons.append("bitsandbytes library is not installed or lacks Windows CPU runtime support.")
        
    return {
        "supported": supported,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "device_name": device_name,
        "bitsandbytes_installed": bnb_installed,
        "bitsandbytes_version": bnb_version,
        "bitsandbytes_error": bnb_error,
        "reasons_for_limitation": reasons,
        "status": "Hardware Limitation: Standard FP32 LoRA is used as primary working pipeline."
    }


def get_qlora_config() -> Dict[str, Any]:
    """
    Return the standard QLoRA BitsAndBytes configuration used on CUDA environments.
    
    Conceptual Specification:
    - load_in_4bit: True (loads base model weights in 4-bit precision)
    - bnb_4bit_quant_type: 'nf4' (NormalFloat4, information-theoretically optimal for normal distributions)
    - bnb_4bit_use_double_quant: True (quantizes the quantization constants, saving ~0.37 bits/param)
    - bnb_4bit_compute_dtype: torch.bfloat16 or torch.float16 (dtype for matrix multiplication in forward pass)
    """
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    }


QLORA_THEORY = """
=== QLoRA: Quantized Low-Rank Adaptation ===

1. Background & Motivation:
   Full fine-tuning requires 16-bit or 32-bit storage for weights, gradients, optimizer states,
   and activations. Standard LoRA reduces trainable parameters and optimizer memory by freezing
   the base model in 16-bit precision and only training low-rank adapter matrices A and B.
   QLoRA (Dettmers et al., 2023) goes further by quantizing the frozen base model weights
   down to 4-bit precision, reducing memory footprint by ~75% for the base model.

2. Key Innovations in QLoRA:
   a. 4-bit NormalFloat (NF4):
      Pretrained neural network weights typically follow a normal distribution N(0, sigma^2).
      Standard integer quantization (Int4) distributes bins uniformly, losing precision in tails.
      NF4 constructs quantile bins such that each bin has an equal probability under a normal
      distribution, minimizing information loss.
      
   b. Double Quantization (DQ):
      Quantization requires scaling constants c_1 to map FP32/16 values to 4-bit integers.
      Double quantization quantizes the quantization constants themselves from 32-bit floats
      to 8-bit integers with 256-block sizes, saving approximately 0.37 bits per parameter.
      
   c. Paged Optimizers:
      Utilizes CUDA Unified Memory to automatically page optimizer states between GPU VRAM
      and CPU RAM during memory spikes (e.g., long sequences), preventing Out-Of-Memory (OOM) errors.

3. Why LoRA Adapters Can Still Be Trained:
   The forward pass computes:
       Y = (dequantize(W_4bit) @ X) + (B @ A @ X) * (alpha / r)
   - W_4bit remains completely frozen in memory.
   - During the forward pass, blocks of W are dequantized on the fly to compute_dtype (BF16/FP16)
     for tensor contraction with activation X.
   - During the backward pass, gradients dL/dY are backpropagated into adapter matrices A and B.
   - No gradients are computed or stored for W (requires_grad = False).
   - Only A and B (which are stored in FP32/BF16) receive gradients and get updated by AdamW.

4. Difference between LoRA and QLoRA:
   +----------------------+---------------------------+---------------------------+
   | Property             | Standard LoRA             | QLoRA                     |
   +----------------------+---------------------------+---------------------------+
   | Base Model Weights   | FP16 or FP32 (16/32 bits) | NF4 (4 bits)              |
   | Adapter Weights      | FP16 or FP32              | FP16 or FP32              |
   | Memory per Base Param| 2 to 4 bytes              | ~0.5 bytes                |
   | Compute Hardware     | CPU or GPU                | Requires CUDA GPU         |
   | Training Throughput  | Higher (no dequant step)  | Slightly slower (dequant) |
   +----------------------+---------------------------+---------------------------+
"""
