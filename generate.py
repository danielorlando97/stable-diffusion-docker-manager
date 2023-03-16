from diffusers import StableDiffusionPipeline
import torch
import sys

model_id = "./textual_inversion_target"
pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")

prompt = sys.argv[1]

image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("image.png")