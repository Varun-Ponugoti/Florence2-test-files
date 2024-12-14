from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, PreTrainedTokenizerFast, BertTokenizer, QuantoConfig
from optimum.gptq import GPTQQuantizer
import torch
import torch.quantization as quantization


# Load the model and processor
device = "cpu"  # Quantization is CPU-friendly
model = AutoModelForCausalLM.from_pretrained("/home/varun/Documents/florence2/Florence-2-large", trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("/home/varun/Documents/florence2/Florence-2-large", trust_remote_code=True)


if not hasattr(model.config, "use_cache"):
    model.config.use_cache = False

# Apply dynamic quantization
quantization_config = QuantoConfig(weights="int8")

quantized_model = AutoModelForCausalLM.from_pretrained(
    "/home/varun/Documents/florence2/Florence-2-large",
    quantization_config= quantization_config
)

model.qconfig = quantization.get_default_qconfig('fbgemm')  # For x86 CPUs
quantized_model = quantization.prepare(model)
quantized_model = quantization.convert(quantized_model)

save_dir = "/home/varun/Documents"
#quantized_model.save_quanitzed("quantized_model", safetensors=True)
#quantized_model.save_pretrained("florence2-quantized")
print("Quantized model saved!")



