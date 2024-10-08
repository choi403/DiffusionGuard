import math
import torch
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as T
from typing import Tuple, Optional, List

totensor = T.ToTensor()
topil = T.ToPILImage()

def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    numpy_array = (tensor + 1.0) * 127.5
    numpy_array = numpy_array.clamp(0, 255).byte().numpy()
    
    if numpy_array.ndim == 4:
        numpy_array = numpy_array.squeeze(0)

    if numpy_array.shape[0] == 1:
        numpy_array = numpy_array.squeeze(0)
        pil_image = Image.fromarray(numpy_array, mode='L')
    elif numpy_array.shape[0] == 3:
        numpy_array = numpy_array.transpose(1, 2, 0)
        pil_image = Image.fromarray(numpy_array, mode='RGB')
    else:
        raise ValueError("Unsupported number of channels: {}".format(numpy_array.shape[0]))
    
    return pil_image


def recover_image(
    image: Image.Image, 
    init_image: Image.Image, 
    mask: Image.Image, 
    background: bool = False
) -> Image.Image:
    image = totensor(image)
    mask = totensor(mask)
    init_image = totensor(init_image)
    if background:
        result = mask * init_image + (1 - mask) * image
    else:
        result = mask * image + (1 - mask) * init_image
    return topil(result)


def preprocess(image: Image.Image) -> torch.Tensor:
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def pil_image_to_tensor(image: Image.Image) -> torch.Tensor:
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image


def prepare_mask_and_masked_image(
    image: Optional[Image.Image], 
    mask: Optional[Image.Image] = None
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if image is not None:
        image = np.array(image.convert("RGB"))
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    else:
        image = None
    
    if mask is None:
        return None, None, image

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    if image is not None:
        masked_image = image * (mask < 0.5)
    else:
        masked_image = None

    return mask, masked_image, image


def invert_and_return_as_tensor(pil_image: Image.Image, size: int = 512) -> torch.Tensor:
    mask_image = ImageOps.invert(pil_image).resize((size, size))
    mask_tensor, _, _ = prepare_mask_and_masked_image(None, mask_image)
    return 1 - mask_tensor


def _overlay_images(image1: Image.Image, image2: Image.Image) -> Image.Image:
    image2 = image2.resize(image1.size)
    result_image = Image.new("RGB", image1.size)

    for x in range(image1.width):
        for y in range(image1.height):
            pixel1 = image1.getpixel((x, y))
            pixel2 = image2.getpixel((x, y))

            new_pixel = tuple(max(p1, p2) for p1, p2 in zip(pixel1, pixel2))

            result_image.putpixel((x, y), new_pixel)

    return result_image


def overlay_images(images: List[Image.Image]) -> Image.Image:
    if len(images) == 1:
        return images[0]
    elif not images:
        raise ValueError("list is empty")
    
    result_image = images[0]

    for image in images[1:]:
        result_image = _overlay_images(result_image, image)

    return result_image


def get_mask_radius_list(mask_list: List[Image.Image]) -> List[float]:
    radius_list = []
    for pil_image in mask_list:
        image_array = np.array(pil_image)
        binary_array = image_array > 128 
        area = np.sum(binary_array)
        radius = (area / (3 * 2 / math.pi)) ** 0.5
        radius_list.append(radius)
    return radius_list
