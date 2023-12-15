import numpy as np
from PIL import Image
import torch
import rp

def make_mask_square( alpha_mask: np.ndarray, method='crop'):
        height, width = rp.get_image_dimensions(alpha_mask)
        min_dim = min(height, width)
        if method == 'crop':
            return make_mask_square(rp.crop_image(alpha_mask, min_dim, min_dim, origin='center'), 'scale')
        if method == 'scale':
            return torch.tensor(rp.resize_image(alpha_mask, (512, 512))).unsqueeze(0).repeat(1, 1, 1)


def calc_iou(mat1,mat2):
    intersection = np.logical_and(mat1.numpy(), mat2.numpy())
    union = np.logical_or(mat1.numpy(), mat2.numpy())

    # Compute IoU
    iou = np.sum(intersection) / np.sum(union)

    print("IoU:", iou)
    return iou