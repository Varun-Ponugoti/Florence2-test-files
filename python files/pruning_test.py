import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import json
import tensorflow as tf
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, PreTrainedTokenizerFast, BertTokenizer, QuantoConfig, BitsAndBytesConfig
from optimum.gptq import GPTQQuantizer
import torch
import torch.quantization as quantization
from torchsummary import summary
from torch import nn
from torch.nn.utils import prune

# Load Florence-2 Model
model_path = "/home/varun/Documents/florence2/Florence-2-large"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)

# Example: Apply pruning to a specific layer (unstructured pruning)
def prune_model_layer(layer, amount=0.3):
    """
    Applies unstructured pruning to a layer with a given sparsity amount.
    """
    if isinstance(layer, nn.Linear):
        prune.l1_unstructured(layer, name="weight", amount=amount)
        print(f"Pruned Layer: {layer}")

# Recursively apply pruning to layers
def prune_model(model, amount=0.3):
    """
    Apply pruning to all linear layers in the model.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):  # Prune Linear layers
            prune_model_layer(module, amount)

# Apply pruning with sparsity level (e.g., 30%)
prune_model(model, amount=0.3)

# Check sparsity
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"{name} - Sparsity: {100. * float(torch.sum(module.weight == 0)) / module.weight.nelement()}%")

# Remove pruning reparametrization for deployment
for module in model.modules():
    if isinstance(module, nn.Linear):
        prune.remove(module, "weight")

# Save pruned model
output_dir = "./pruned_florence2"
model.save_pretrained(output_dir)
print(f"Pruned model saved to {output_dir}")

