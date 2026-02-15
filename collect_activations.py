"""
collect_activations.py - hook into qwen and save residual stream activations
"""

import modal

app = modal.App("qwen-activation-collector")
MODEL_ID = "Qwen/Qwen2.5-1.5B"

# storage for activations
activation_storage = modal.Volume.from_name("activation-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
    )
)

@app.cls(
    gpu="a10g",
    image=image,
    timeout=3600,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/activations": activation_storage,
    },
)
class ActivationCollector:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        
        # storing the collected activations
        self.captured_activations = []
    
    @modal.method()
    def collect_from_text(self, text: str, target_layer: int = 14):
        """
        run text through model and capture residual stream at target_layer
        
        theory: we're inserting a "hook" that grabs the activation tensor
        as it flows through the layer, without changing the forward pass
        """
        import torch

        def hook_fn(module, input, output):
            # output[0] is the residual stream tensor
            # shape: [batch_size, sequence_length, hidden_size]
            self.captured_activations.append(output[0].detach().cpu())
        
        # register the hook at the target layer
        # qwen architecture: model.layers[i] is each transformer block
        handle = self.model.model.layers[target_layer].register_forward_hook(hook_fn)
        
        # running the forward pass
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # cleaning up the hook
        handle.remove()
        
        if self.captured_activations:
            act = self.captured_activations[-1]
            return {
                "shape": list(act.shape),
                "layer": target_layer,
                "num_tokens": act.shape[1],
            }
        return None


@app.local_entrypoint()
def main():
    collector = ActivationCollector()
    
    # testing with one example for now
    test_text = "人工智能的未来充满希望"
    
    result = collector.collect_from_text.remote(test_text, target_layer=14)
    print("captured activation info:", result)
