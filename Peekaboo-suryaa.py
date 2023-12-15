import rp
import source.peekaboo as peekaboo
from source.peekaboo import run_peekaboo
from source.clip import get_clip_logits


def run_inference(image_name, caption):
    
    run_peekaboo(
                    caption,
                     f"/home/ubuntu/Detic/detic_input/{image_name}.jpg",
                     f"/home/ubuntu/Detic/detic_output/{image_name}.pt",
                     # "/home/ubuntu/Detic/detic_op/final_mask_rgb.png",
                     representation='detic', LEARNING_RATE= 0.01, #LEARNING_RATE=1e-04, 
                     GUIDANCE_SCALE=300, GRAVITY=0.08,
                     # GUIDANCE_SCALE=100, GRAVITY=.05,
                     min_step=150,
                     max_step=500,
                     NUM_ITER=200,
                    bilateral_kwargs={'iterations': 80, 'kernel_size': 3, 'sigma': 5, 'tolerance': 0.08}
                     )
    

image_name = [5048, 5570, 10916, 11595, 12097]
captions = ["yellow shirt on the right", "the girl on the left", "man in orange shirt", "guy on the left with purple shirt", "black shirt guy in the centre of photo"]

for i in range(len(image_name)):
    run_inference(image_name=image_name[i], caption=captions[i])
