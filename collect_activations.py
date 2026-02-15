"""
    collect_activations.py:
    collecting activations from a model using a hooking function
"""

import torch


def collect_from_text(model, tokenizer, text: str, target_layer: int = 14):
    """Run *text* through *model* and capture the residual stream at *target_layer*.

    Args:
        model:        a loaded HuggingFace causal-lm (e.g. Qwen2.5-1.5B)
        tokenizer:    the matching tokenizer
        text:         raw string to feed through the model
        target_layer: which transformer block to hook (0-indexed)

    Returns:
        dict with shape, layer, num_tokens  –  or None if nothing was captured
    """
    captured = []

    def hook_fn(module, input, output):
        # output[0] is the residual stream tensor
        # shape: [batch_size, sequence_length, hidden_size]
        captured.append(output[0].detach().cpu())

    # qwen architecture: model.model.layers[i] is each transformer block
    handle = model.model.layers[target_layer].register_forward_hook(hook_fn)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model(**inputs)

    handle.remove()

    if captured:
        act = captured[-1]
        return {
            "activation": act,
            "shape": list(act.shape),
            "layer": target_layer,
            "num_tokens": act.shape[1],
        }
    return None
