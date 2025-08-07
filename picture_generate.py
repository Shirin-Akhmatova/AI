from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)
pipe = pipe.to("cpu")

prompt = (
    "Tiny cute dragon with glowing wings sitting on a human hand, sunny daylight, "
    "highly detailed, fantasy style, soft focus, photorealistic, concept art"
)
image = pipe(prompt).images[0]

image.save("dragon.png")
