import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import json
import tensorflow as tf
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, PreTrainedTokenizerFast, BertTokenizer, QuantoConfig, BitsAndBytesConfig
from optimum.gptq import GPTQQuantizer
import torch
import torch.quantization as quantization

device = "cpu"  # Quantization is CPU-friendly
model = AutoModelForCausalLM.from_pretrained("/home/varun/Documents/florence2/Florence-2-large", trust_remote_code=True).to(device)

bnb_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model=AutoModelForCausalLM.from_pretrained("/home/varun/Documents/florence2/Florence-2-large",
    quantization_config=bnb_config,
    device_map='auto'
)

print(model.get_memory_footprint())
