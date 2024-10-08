from .common import generate_perturbation
from .attack_diffusionguard import attack_pipeline as diffusionguard_attack_pipeline

from typing import List, Dict, Any
from PIL import Image
from torch import Tensor
from diffusers import StableDiffusionPipeline
from omegaconf import DictConfig


def protect_image(
    method: str,
    pipe: StableDiffusionPipeline,
    src_image: Image.Image,
    mask_image_list: List[Image.Image],
    mask_image_combined: Image.Image,
    mask_radius_list: List[float],
    config: DictConfig
) -> Tensor:
    '''
    Applies a protection method to the source image using specified parameters.

    Args:
    method (str): Protection method to protect the given image.
    pipe (StableDiffusionPipeline): A diffusers pipeline object to protect the image against.
    src_image (Image.Image): Source image to protect. Adversarial perturbation will be added.
    mask_image_list (List[Image.Image]): List of mask pieces for masked inpainting.
    mask_image_combined (Image.Image): The combined image (Union) of the list of mask pieces.
    mask_radius_list (List[float]): List of mask sizes defined as the radii of circles of the 
    same area.
    config (DictConfig): Config given by main.py.
    
    Returns:
    Tensor: The resulting protected image that includes the adversarial perturbation added.
    '''
    if method == 'diffusionguard':
        return generate_perturbation(
            diffusionguard_attack_pipeline,
            pipe, 
            src_image, 
            mask_image_list,
            mask_image_combined,
            mask_radius_list,
            config
        )
    else:
        raise NotImplementedError()
