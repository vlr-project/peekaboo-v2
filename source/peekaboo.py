from typing import Union, List, Optional

import icecream
import numpy as np
import rp
import torch
import torch.nn as nn
from PIL.Image import Image
from easydict import EasyDict
from IPython.display import clear_output
from torchvision.transforms.functional import normalize

import source.stable_diffusion as sd
from source.bilateral_blur import BilateralProxyBlur
from source.learnable_textures import (LearnableImageFourier,
                                       LearnableImageFourierBilateral,
                                       LearnableImageRaster,
                                       LearnableImageRasterBilateral,
                                       LearnableTexturePackFourier,
                                       LearnableTexturePackRaster,
                                       LearnableImageRasterBilateralDetic,
                                       LearnableImageRasterDetic)


def make_learnable_image(height, width, num_channels, foreground=None, bilateral_kwargs: dict = {},
                         representation='fourier', init_alpha=None):
    # Here we determine our image parametrization schema
    bilateral_blur = BilateralProxyBlur(foreground, **bilateral_kwargs)
    if representation == 'fourier bilateral':
        return LearnableImageFourierBilateral(bilateral_blur, num_channels)  # A neural neural image + bilateral filter
    elif representation == 'raster bilateral':
        return LearnableImageRasterBilateral(bilateral_blur, num_channels)  # A regular image + bilateral filter
    elif representation == 'fourier':
        return LearnableImageFourier(height, width, num_channels)  # A neural neural image
    elif representation == 'raster':
        return LearnableImageRaster(height, width, num_channels)  # A regular image
    elif representation == 'detic':
        # return LearnableImageRasterDetic(init_alpha, height, width, num_channels,bilateral_blur) # Regular images using detic bounding boxes
        return LearnableImageRasterBilateralDetic(init_alpha, height, width, num_channels, bilateral_blur)
    else:
        assert False, 'Invalid method: ' + representation


def blend_torch_images(foreground, background, alpha):
    # Input assertions
    assert foreground.shape == background.shape
    C, H, W = foreground.shape
    assert alpha.shape == (H, W), f'alpha is a matrix {alpha.shape} != ({H}, {W})'

    return foreground * alpha + background * (1 - alpha)


class PeekabooSegmenter(nn.Module):
    def __init__(self,
                 image: np.ndarray,
                 original_image,
                 bb_path,
                 labels: List['BaseLabel'],
                 size: int = 512,
                 name: str = 'Untitled',
                 bilateral_kwargs: dict = {},
                 representation='fourier bilateral',
                 min_step=None,
                 max_step=None,
                 ):

        super().__init__()
        
        height = width = size  # We use square images for now

        assert all(issubclass(type(label), BaseLabel) for label in labels)
        assert len(labels), 'Must have at least one class to segment'
        self.original_image = original_image
        self.height = height
        self.bb_path = bb_path
        self.width = width
        self.labels = labels
        self.name = name
        self.representation = representation
        self.min_step = min_step
        self.max_step = max_step

        assert rp.is_image(image), 'Input should be a numpy image'
        image = rp.cv_resize_image(image, (height, width))
        image = rp.as_rgb_image(image)  # Make sure it has 3 channels in HWC form
        image = rp.as_float_image(image)  # Make sure it's values are between 0 and 1
        assert image.shape == (height, width, 3) and image.min() >= 0 and image.max() <= 1
        self.image = image

        self.foreground = rp.as_torch_image(image).to(device)  # Convert the image to a torch tensor in CHW form
        assert self.foreground.shape == (3, height, width)

        self.background = self.foreground * 0  # The background will be a solid color for now

        original_alpha_mask = self.get_alpha_mask().numpy()
        init_alpha = self.make_mask_square(original_alpha_mask)
        self.alphas = make_learnable_image(height, width, num_channels=self.num_labels, init_alpha=init_alpha, foreground=self.foreground,
                                           representation=self.representation, bilateral_kwargs=bilateral_kwargs)

    @property
    def num_labels(self):
        return len(self.labels)

    def set_background_color(self, color):
        r, g, b = color
        assert 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1
        self.background[0] = r
        self.background[1] = g
        self.background[2] = b

    def randomize_background(self):
        self.set_background_color(rp.random_rgb_float_color())

    def get_alpha_mask(self):
        bounding_boxes = torch.load(self.bb_path).detach().cpu()
        
        height, width = rp.get_image_dimensions(self.original_image)

        mask_tensor = torch.zeros((height, width))

        for box in bounding_boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            mask_tensor[y_min:y_max, x_min:x_max] = 1

        return mask_tensor #.unsqueeze(-1).expand_as(torch.tensor(self.original_image))

    def make_mask_square(self, alpha_mask: np.ndarray, method='crop'):
        height, width = rp.get_image_dimensions(alpha_mask)
        min_dim = min(height, width)
        if method == 'crop':
            return self.make_mask_square(rp.crop_image(alpha_mask, min_dim, min_dim, origin='center'), 'scale')
        if method == 'scale':
            return torch.tensor(rp.resize_image(alpha_mask, (512, 512))).unsqueeze(0).repeat(self.num_labels, 1, 1) #.unsqueeze(-1).expand((512, 512, 3))


    def forward(self, alphas=None, bbox=None,return_alphas=False):
        try:
            old_min_step = s.min_step
            old_max_step = s.max_step
            s.min_step = self.min_step
            s.max_step = self.max_step

            output_images = []

            if alphas is None:
                alphas = self.alphas()

            alphas = torch.clamp(alphas, min=0., max=1)
            assert alphas.shape == (self.num_labels, self.height, self.width)
            assert alphas.min() >= 0 and alphas.max() <= 1, f"min = {alphas.min()}, max={alphas.max()}"

            for alpha in alphas:
                output_image = blend_torch_images(foreground=self.foreground, background=self.background, alpha=alpha)
                output_images.append(output_image)

            output_images = torch.stack(output_images)
            # output_images = blend_torch_images(foreground=self.foreground,background=self.background,alpha=alphas)
            assert output_images.shape == (self.num_labels, 3, self.height, self.width)  # In BCHW form

            if return_alphas:
                return output_images, alphas
            else:
                return output_images

        finally:
            old_min_step = s.min_step
            old_max_step = s.max_step


def display(self):
    # This is a method of PeekabooSegmenter, but can be changed without rewriting the class if you want to change the display

    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), ]  # (1,0,0), (0,1,0), (0,0,1)] #Colors used to make the display
    colors = [rp.random_rgb_float_color() for _ in range(3)]
    alphas = rp.as_numpy_array(self.alphas())
    image = self.image
    assert alphas.shape == (self.num_labels, self.height, self.width)

    composites = []
    for color in colors:
        self.set_background_color(color)
        column = rp.as_numpy_images(self(self.alphas()))
        composites.append(column)

    label_names = [label.name for label in self.labels]

    stats_lines = [
        self.name,
        '',
        'H,W = %ix%i' % (self.height, self.width),
    ]

    def try_add_stat(stat_format, var_name):
        if var_name in globals():
            stats_line = stat_format % globals()[var_name]
            stats_lines.append(stats_line)

    try_add_stat('Gravity: %.2e', 'GRAVITY')
    try_add_stat('Batch Size: %i', 'BATCH_SIZE')
    try_add_stat('Iter: %i', 'iter_num')
    try_add_stat('Image Name: %s', 'image_filename')
    try_add_stat('Learning Rate: %.2e', 'LEARNING_RATE')
    try_add_stat('Guidance: %i%%', 'GUIDANCE_SCALE')

    stats_image = rp.labeled_image(self.image, rp.line_join(stats_lines),
                                   size=15 * len(stats_lines),
                                   position='bottom', align='center')

    composite_grid = rp.grid_concatenated_images([
        rp.labeled_images(alphas, label_names),
        *composites
    ])

    assert rp.is_image(self.image)
    assert rp.is_image(alphas[0])
    assert rp.is_image(composites[0][0])
    assert rp.is_image(composites[1][0])
    assert rp.is_image(composites[2][0])

    output_image = rp.labeled_image(
        rp.tiled_images(
            rp.labeled_images(
                [
                    self.image,
                    alphas[0],
                    composites[0][0],
                    composites[1][0],
                    composites[2][0],
                ],
                [
                    "Input Image",
                    "Alpha Map",
                    "Background #1",
                    "Background #2",
                    "Background #3",
                ],
            ),
            length=2 + len(composites),
        ),
        label_names[0],
    )

    # output_image = rp.horizontally_concatenated_images(stats_image, composite_grid)

    rp.display_image(output_image)

    return output_image


PeekabooSegmenter.display = display


def get_mean_embedding(prompts: list):
    return torch.mean(
        torch.stack(
            [s.get_text_embeddings(prompt) for prompt in prompts]
        ),
        dim=0
    ).to(device)


class BaseLabel:
    def __init__(self, name: str, embedding: torch.Tensor):
        # Later on we might have more sophisticated embeddings, such as averaging multiple prompts
        # We also might have associated colors for visualization, or relations between labels
        self.name = name
        self.embedding = embedding

    def get_sample_image(self):
        output = s.embeddings_to_imgs(self.embedding)[0]
        assert rp.is_image(output)
        return output

    def __repr__(self):
        return '%s(name=%s)' % (type(self).__name__, self.name)


class SimpleLabel(BaseLabel):
    def __init__(self, name: str):
        super().__init__(name, s.get_text_embeddings(name).to(device))


class MeanLabel(BaseLabel):
    # Test: rp.display_image(rp.horizontally_concatenated_images(MeanLabel('Dogcat','dog','cat').get_sample_image() for _ in range(1)))
    def __init__(self, name: str, *prompts):
        prompts = rp.detuple(prompts)
        super().__init__(name, get_mean_embedding(prompts))


def log_cell(cell_title):
    rp.fansi_print("<Cell: %s>" % cell_title, 'cyan', 'underlined')
    # rp.ptoc()


def log(x):
    x = str(x)
    rp.fansi_print(x, 'yellow')


class PeekabooResults(EasyDict):
    # Acts like a dict, except you can read/write parameters by doing self.thing instead of self['thing']
    pass


def save_peekaboo_results(results, new_folder_path):
    assert not rp.folder_exists(new_folder_path), 'Please use a different name, not %s' % new_folder_path
    rp.make_folder(new_folder_path)
    with rp.SetCurrentDirectoryTemporarily(new_folder_path):
        log("Saving PeekabooResults to " + new_folder_path)
        params = {}
        for key in results:
            value = results[key]
            if rp.is_image(value):
                # Save a single image
                rp.save_image(value, key + '.png')
            elif isinstance(value, np.ndarray) and rp.is_image(value[0]):
                # Save a folder of images
                rp.make_directory(key)
                with rp.SetCurrentDirectoryTemporarily(key):
                    for i in range(len(value)):
                        rp.save_image(value[i], str(i) + '.png')
            elif isinstance(value, np.ndarray):
                # Save a generic numpy array
                np.save(key + '.npy', value)
            else:

                import json
                try:
                    json.dumps({key: value})
                    # Assume value is json-parseable
                    params[key] = value
                except Exception:
                    params[key] = str(value)
        rp.save_json(params, 'params.json', pretty=True)
        log("Done saving PeekabooResults to " + new_folder_path + "!")


def make_image_square(image: np.ndarray, method='crop') -> np.ndarray:
    # Takes any image and makes it into a 512x512 square image with shape (512,512,3)
    assert rp.is_image(image)
    assert method in ['crop', 'scale']
    image = rp.as_rgb_image(image)

    height, width = rp.get_image_dimensions(image)
    min_dim = min(height, width)
    max_dim = max(height, width)

    if method == 'crop':
        return make_image_square(rp.crop_image(image, min_dim, min_dim, origin='center'), 'scale')
    if method == 'scale':
        return rp.resize_image(image, (512, 512))

def print_grad_hook(grad):
    print("Gradient:", grad)

def run_peekaboo(name: str, image: Union[str, np.ndarray], bounding_box_path: str, label: Optional['BaseLabel'] = None,
                 
                 # Peekaboo Hyperparameters:
                 GRAVITY=1e-1 / 2,  # This is the one that needs the most tuning, depending on the prompt...
                 #   ...usually one of the following GRAVITY will work well: 1e-2, 1e-1/2, 1e-1, or 1.5*1e-1
                 NUM_ITER=300,  # 300 is usually enough
                 LEARNING_RATE=1e-5,
                 # Can be larger if not using neural neural textures (aka when representation is raster)
                 BATCH_SIZE=1,  # Doesn't make much difference, larger takes more vram
                 GUIDANCE_SCALE=100,  # The defauly value from the DreamFusion paper
                 bilateral_kwargs=dict(kernel_size=3,
                                       tolerance=.08,
                                       sigma=5,
                                       iterations=40,
                                       ),
                 square_image_method='crop',  # Can be either 'crop' or 'scale' - how will we square the input image?
                 representation='fourier bilateral',
                 # Can be 'fourier bilateral', 'raster bilateral', 'fourier', or 'raster'
                 min_step=None,
                 max_step=None,
                 ) -> PeekabooResults:
    if label is None:
        label = SimpleLabel(name)

    
    image_path = '<No image path given>'
    if isinstance(image, str):
        image_path = image
        image = rp.load_image(image)

    assert rp.is_image(image)

    assert issubclass(type(label), BaseLabel)

    original_image = image.copy()

    image = rp.as_rgb_image(rp.as_float_image(make_image_square(image, square_image_method)))
    rp.tic()
    time_started = rp.get_current_date()

    log_cell('Get Hyperparameters')  ########################################################################
    icecream.ic(GRAVITY, BATCH_SIZE, NUM_ITER, LEARNING_RATE, GUIDANCE_SCALE, representation, bilateral_kwargs,
                square_image_method)

    # log_cell('Alpha Initializer') ########################################################################

    p = PeekabooSegmenter(image,
                          original_image,
                          bounding_box_path,
                          labels=[label],
                          name=name,
                          bilateral_kwargs=bilateral_kwargs,
                          representation=representation,
                          min_step=min_step,
                          max_step=max_step,
                          ).to(device)

    if 'bilateral' in representation:
        blur_image = rp.as_numpy_image(p.alphas.bilateral_blur(p.foreground))
        print("The bilateral blur applied to the input image before/after, to visualize it")
        rp.display_image(
            rp.tiled_images(rp.labeled_images([rp.as_numpy_image(p.foreground), blur_image], ['before', 'after'])))

    p.display();

    # log_cell('Create Optimizers') ########################################################################

    params = list(p.parameters())
    print(params)
    optim = torch.optim.Adam(params, lr=1e-3)
    optim = torch.optim.SGD(params, lr=LEARNING_RATE)

    # log_cell('Create Logs') ########################################################################
    global iter_num
    iter_num = 0
    timelapse_frames = []

    # log_cell('Do Training') ########################################################################
    preview_interval = NUM_ITER // 10  # Show 10 preview images throughout training to prevent output from being truncated
    preview_interval = max(1, preview_interval)
    log("Will show preview images every %i iterations" % (preview_interval))
    gravity_decay_interval = NUM_ITER // 10
    current_gravity = GRAVITY
    gravity_decay_rate = 0.95
    L2_REGULARIZATION = 0.005
    try:
        display_eta = rp.eta(NUM_ITER)
        for i in range(NUM_ITER):
            display_eta(i)
            iter_num += 1

            alphas = p.alphas()
            # param_hook = alphas.register_hook(print_grad_hook)

            for __ in range(BATCH_SIZE):
                p.randomize_background()
                composites = p()
                for label, composite in zip(p.labels, composites):
                    s.train_step(label.embedding, composite[None],
                                 guidance_scale=GUIDANCE_SCALE
                                 )
            if i % gravity_decay_interval == 0 and i != 0:
                current_gravity *= gravity_decay_rate
                print(f"Updated Gravity: {current_gravity}")

            # ((alphas.sum()) * GRAVITY).backward()
            l2_reg = L2_REGULARIZATION * (alphas ** 2).sum()

            # Calculate total loss with L2 regularization
            total_loss = ((alphas.sum()) * current_gravity) + l2_reg

            total_loss.backward()
            
            optim.step()
            optim.zero_grad()

            with torch.no_grad():
                # if not _%100:
                # Don't overflow the notebook
                # clear_output()
                if not i % preview_interval:
                    timelapse_frames.append(p.display())
                    # rp.ptoc()
            # param_hook.remove()

    except KeyboardInterrupt:
        log("Interrupted early, returning current results...")
        pass

    # rp.ptoc()
    results = PeekabooResults(
        # The main output is the alphas
        alphas=rp.as_numpy_array(alphas),

        # Keep track of hyperparameters used
        GRAVITY=GRAVITY,
        BATCH_SIZE=BATCH_SIZE,
        NUM_ITER=NUM_ITER,
        GUIDANCE_SCALE=GUIDANCE_SCALE,
        bilateral_kwargs=bilateral_kwargs,
        representation=representation,

        # Keep track of the inputs used
        label=label,
        image=image,
        image_path=image_path,

        # Record some extra info
        preview_image=p.display(),
        timelapse_frames=rp.as_numpy_array(timelapse_frames),
        **({'blur_image': blur_image} if 'blur_image' in dir() else {}),
        height=p.height,
        width=p.width,
        p_name=p.name,

        min_step=p.min_step,
        max_step=p.max_step,

        # git_hash=rp.get_current_git_hash(), 
        time_started=rp.r._format_datetime(time_started),
        time_completed=rp.r._format_datetime(rp.get_current_date()),
        device=device,
        computer_name=rp.get_computer_name(),
    )

    output_folder = rp.make_folder('peekaboo_results/%s' % name)
    output_folder += '/%03i' % len(rp.get_subfolders(output_folder))

    save_peekaboo_results(results, output_folder)
    # print("Please wait - creating a training timelapse")
    # clear_output()
    # rp.display_image_slideshow(timelapse_frames)  # This can take a bit of time
    print("Saved results at %s" % output_folder)
    icecream.ic(name, label, image_path, GRAVITY, BATCH_SIZE, NUM_ITER, GUIDANCE_SCALE, bilateral_kwargs)

    return results


# Importing this module loads a stable diffusion model. Hope you have a GPU!
s = sd.StableDiffusion('cuda', 'CompVis/stable-diffusion-v1-4')
device = s.device
