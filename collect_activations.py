"""
collect_activations.py – pure function for capturing residual-stream activations

how it works:
    pytorch lets you attach a "hook" to any layer in a model. when the forward
    pass reaches that layer, pytorch calls your hook function with the layer's
    output tensor *before* passing it to the next layer. the hook can read (or
    even modify) the tensor, but here we only read it.

    so the flow is:
        1. register a hook on model.layers[target_layer]
        2. run the normal forward pass (tokenize → model(**inputs))
        3. as the forward pass hits the target layer, our hook fires and
           copies the residual-stream tensor to cpu
        4. remove the hook so it doesn't fire on future forward passes
        5. return the captured tensor + metadata

    the residual stream at layer L is the hidden state *after* that transformer
    block's attention + mlp, before the next block. its shape is
    [batch_size, sequence_length, hidden_size] – for qwen2.5-1.5b that last
    dim is 1536.

this file has zero modal / gpu dependencies. it's a plain function that takes
a model + tokenizer and returns activation info. the QwenModel class in
qwen_inference.py calls it inside a @modal.method() so it runs on the gpu.
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
