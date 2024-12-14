import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM , AutoConfig

device = "cuda"
torch_dtype = torch.float32

model = AutoModelForCausalLM.from_pretrained("/home/varun/Documents/florence 2/quantized_model_awq", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("/home/varun/Documents/florence 2/quantized_model_awq", trust_remote_code=True)

url = "traffic-1.jpg"
image = Image.open(url)

def run_example(task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

    print(parsed_answer)
    
prompt = "<CAPTION>"
run_example(prompt)
