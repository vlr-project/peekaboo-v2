import source.peekaboo as peekaboo
from source.peekaboo import run_peekaboo
peekaboo.s.min_step, peekaboo.s.max_step=200,600
results = run_peekaboo('Mario',
                     "https://i1.sndcdn.com/artworks-000160550668-iwxjgo-t500x500.jpg",
                     representation='raster bilateral', LEARNING_RATE=1e-0, GUIDANCE_SCALE=200, NUM_ITER=500,  GRAVITY=.05, min_step=200,max_step=900,
                     )

# import torch
# from diffusers import StableDiffusionPipeline
#
# model_id = "CompVis/stable-diffusion-v1-4"
# device = "cuda"
#
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to(device)
#
# prompt = "a photo of a cowboy astronaut riding a horse on mars jumping over water"
# image = pipe(prompt).images[0]
#
# image.save("astronaut_rides_horse.png")