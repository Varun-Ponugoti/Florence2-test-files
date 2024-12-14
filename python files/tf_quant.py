from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

"""
Download https://huggingface.co/liuhaotian/llava-llama-2-13b-chat-lightning-preview to local
Make following edits to the config.json
LlavaLlamaForCausalLM -> LlamaForCausalLM
"model_type": "llava" -> "llama"
"""
pretrained_model_dir = "/home/varun/Documents/florence2/Florence-2-large"


quantize_config = BaseQuantizeConfig(
    bits=4,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad 
)

# load un-quantized model, by default, the model will always be loaded into CPU memory
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config, trust_remote_code=True)

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
#model.quantize(examples)

# save quantized model using safetensors
model.save_quantized(quantized_model_dir, use_safetensors=True)
