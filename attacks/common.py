from utils import pil_image_to_tensor, get_mask
import torch
from PIL import Image
from typing import List, Any
from torch import Tensor
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from omegaconf import DictConfig

def generate_perturbation(
    attack_pipeline: Any,
    pipe: StableDiffusionPipeline,
    src_image: Image.Image,
    mask_image_list: List[Image.Image],
    mask_image_combined: Image.Image,
    mask_radius_list: List[float],
    config: DictConfig,
    adv_init: Tensor = None,
    **kwargs: Any,
) -> Tensor:
    '''
    Given an attack method, an attack pipeline, a source image to protect, set of mask images,
    and the radii (sizes) of the masks, generate a protected image with adversarial perturbation
    added on top of it.
    
    Args:
    attack_pipeline (Any): Function for pipeline attack method. 
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
    src_image_orig = pil_image_to_tensor(src_image).cuda().half()
    adv = src_image_orig.clone()
    if adv_init is not None:
        adv = adv + adv_init.cuda().half()
    pixel_values = src_image_orig.clone()

    it = tqdm(range(config.training.iters))

    torch.set_grad_enabled(True)
    for _ in it:
        mask = get_mask(
            mask_image_list,
            mask_image_combined,
            mask_radius_list,
            mode=config.training.mask.generation_method,
            size=config.training.size,
            contour_strength=config.training.mask.contour_strength,
            contour_smoothness=config.training.mask.contour_smoothness,
            contour_iters=config.training.mask.contour_iters
        ).half().cuda()

        masked_adv = adv * (mask < 0.5)

        grads = []
        losses = []
        for _ in range(config.training.grad_reps):
            # Clone to prevent graph from accumulating
            cur_mask = mask.clone()
            cur_masked_adv = masked_adv.clone()
            cur_mask.requires_grad = False
            cur_masked_adv.requires_grad_()

            loss = attack_pipeline(
                pipe, 
                pixel_values, 
                mask=cur_mask, 
                masked_image=cur_masked_adv, 
                num_inference_steps=config.training.num_inference_steps, 
                batch_size=config.training.batch_size, 
                **kwargs
            )

            grad = torch.autograd.grad(loss, [cur_masked_adv], allow_unused=True)[0] * (1 - cur_mask)

            grads.append(grad)
            losses.append(loss)

        avg_grad = torch.stack(grads).mean(0)
        
        # Update adversarial image
        adv = adv - avg_grad.detach().sign() * config.training.step_size
        # Clip into noise budget
        adv = torch.minimum(torch.maximum(adv, src_image_orig - config.training.eps), src_image_orig + config.training.eps)
        # Clip into image range
        adv.data = torch.clamp(adv, min=-1.0, max=1.0)

        it.set_description_str(f'Loss: {torch.mean(torch.stack(losses)):.8f}')
    
    return adv
