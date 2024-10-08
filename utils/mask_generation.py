import random
import numpy as np
import torch
from PIL import Image, ImageOps
import cv2
from scipy.ndimage import gaussian_filter
from typing import List
from torch import Tensor

from .utils import prepare_mask_and_masked_image, invert_and_return_as_tensor, tensor_to_pil_image

def random_contour_perturbation(
    mask: Tensor, 
    eps: int, 
    sigma: float, 
    contour_iters: int = 15
) -> Tensor:
    '''
    Applies random perturbations to the contours of a binary mask to create a modified mask.

    Args:
    mask (Tensor): A binary mask tensor (single-channel), where foreground is 1 and background is 0.
    eps (int): Maximum for random offsets applied to contour points.
    sigma (float): Standard deviation for the Gaussian filter applied to random offsets for smoothing.
    contour_iters (int, optional): Number of iterations to apply contour perturbations. Defaults to 15.

    Returns:
    Tensor: The modified binary mask tensor after applying contour perturbations.
    '''
    mask_np = mask.cpu().numpy().astype(np.uint8)
    inverted_mask = 1 - mask_np

    iters = int(random.uniform(1, contour_iters))
    for i in range(iters):
        # Find the contours of the face region
        contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any contours were found
        if len(contours) == 0:
            return mask

        # Select the contour with the largest area (key region)
        contour = max(contours, key=cv2.contourArea)
        contour_np = contour.reshape(-1, 2)

        # Random offsets 
        num_points = len(contour_np)
        offsets_x = np.random.randint(-eps, eps + 1, size=num_points)
        offsets_y = np.random.randint(-eps, eps + 1, size=num_points)

        smoothed_offsets_x = gaussian_filter(offsets_x, sigma=sigma)
        smoothed_offsets_y = gaussian_filter(offsets_y, sigma=sigma)

        # Shrink contour
        perturbed_contour = contour_np.copy()
        perturbed_contour[:, 0] += smoothed_offsets_x
        perturbed_contour[:, 1] += smoothed_offsets_y

        # Ensure the perturbed contour stays within image bounds
        perturbed_contour[:, 0] = np.clip(perturbed_contour[:, 0], 0, mask_np.shape[1] - 1)
        perturbed_contour[:, 1] = np.clip(perturbed_contour[:, 1], 0, mask_np.shape[0] - 1)

        # Ensure perturbed points stay within the original mask
        for i in range(num_points):
            x, y = perturbed_contour[i]
            if inverted_mask[int(y), int(x)] == 0:  # Point is outside the mask
                # Find the nearest point on the original contour
                distances = np.sqrt((contour_np[:, 0] - x)**2 + (contour_np[:, 1] - y)**2)
                nearest_idx = np.argmin(distances)
                perturbed_contour[i] = contour_np[nearest_idx]

        # Create a new mask with the same shape as the original mask
        modified_mask = np.zeros_like(inverted_mask)

        perturbed_contour = perturbed_contour.reshape((-1, 1, 2))
        cv2.drawContours(modified_mask, [perturbed_contour.astype(int)], 0, 1, cv2.FILLED)
        inverted_mask = modified_mask

    modified_mask = 1 - modified_mask

    return torch.from_numpy(modified_mask).to(mask.device)


def get_mask(
    mask_image_list: List[Image.Image],
    mask_image_combined: Image.Image,
    mask_radius_list: List[float],
    mode: str,
    size: int = 512,
    contour_strength: float = 1.0,
    contour_smoothness: float = 0.1,
    contour_iters: int = 15
) -> Tensor:
    '''
    Generates a mask tensor based on the provided mask images, mode, and parameters.

    Args:
    mask_image_list (List[Image.Image]): List of individual mask images for each component.
    mask_image_combined (Image.Image): Combined mask image of all components.
    mask_radius_list (List[float]): List of radius values for each mask component, used for perturbations.
    mode (str): Mode of mask generation ('single', 'global', 'contour_shrink').
    size (int, optional): Size to which masks are resized. Default is 512.
    contour_strength (float, optional): Scaling factor for perturbation distance in 'contour_shrink' mode. Default is 1.0.
    contour_smoothness (float, optional): Standard deviation for the Gaussian filter applied to perturbations in 'contour_shrink' mode. Default is 0.1.
    contour_iters (int, optional): Number of iterations for contour perturbation in 'contour_shrink' mode. Default is 15.

    Returns:
    Tensor: A mask tensor suitable for use in inpainting or other image processing tasks.
    '''
    if mode == 'single':
        # Single fixed mask
        return invert_and_return_as_tensor(
            ImageOps.invert(mask_image_combined.convert('L')), size=size)
    elif mode == 'global':
        # No mask (noise added to the entire image)
        return invert_and_return_as_tensor(
            ImageOps.invert(Image.new('RGB', (size, size), 'white')), size=size)
    elif mode == 'contour_shrink':
        train_mask_pil_list = mask_image_list
        # Contour shrink algorithm in DiffusionGuard
        arrays = []
        for i, train_mask_pil_component in enumerate(train_mask_pil_list):
            cur_mask_radius = mask_radius_list[i]
            perturb_dist = random.uniform(0, cur_mask_radius)
            mask_segment, _, _ = prepare_mask_and_masked_image(
                None, 
                (ImageOps.invert(train_mask_pil_component)).resize((size, size))
            )
            mask_image = random_contour_perturbation(
                mask=mask_segment.squeeze(0).squeeze(0), 
                eps=int(perturb_dist * contour_strength), 
                sigma=cur_mask_radius * contour_smoothness,
                contour_iters=contour_iters
            ).unsqueeze(0)
            
            mask_image = tensor_to_pil_image(mask_image).convert('RGB')

            arr = np.array(mask_image)
            arrays.append(arr)

        combined_arr = arrays[0]
        for arr in arrays[1:]:
            combined_arr = np.minimum(combined_arr, arr)

        mask_image = Image.fromarray(combined_arr, 'RGB')

        return invert_and_return_as_tensor(mask_image, size=size)
    else:
        raise NotImplementedError()
