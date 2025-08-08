from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)
pipe = pipe.to("cpu")

prompt = (  
    "Genshin Impact electro ninja, purple and black kimono, "  
    "lightning patterns on sleeves, katana crackling with energy, "  
    "dynamic mid-air pose, sakura petals swirling around, "  
    "Inazuma city lights in background, anime cel-shaded"  
)
image = pipe(prompt).images[0]

image.save("electro_ninja.png")
