from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

from attacks import protect_image
from utils import overlay_images, get_mask_radius_list, tensor_to_pil_image

import hydra
from omegaconf import DictConfig, OmegaConf
import os

@hydra.main(version_base=None, config_path="config", config_name=None)
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        config.model.inpainting,
        variant="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")

    orig_image_path = os.path.join('assets', config.orig_image_name)
    mask_image_paths = [os.path.join('assets', i) for i in config.mask_image_names]

    src_image = Image.open(orig_image_path)
    mask_image_list = [Image.open(p).convert('RGB').resize((512, 512)) for p in mask_image_paths]

    mask_image_combined = overlay_images(mask_image_list)
    mask_radius_list = get_mask_radius_list(mask_image_list)

    output_exp_folder = "./protected_images"
    output_exp_folder += f"/{config.exp_name}"
    output_img_folder = output_exp_folder + f"/{config.orig_image_name.split('.png')[0]}/"
    os.makedirs(output_img_folder, exist_ok=True)

    # Save the config file in the experiment directory
    config_save_path = os.path.join(output_exp_folder, "config.yaml")
    with open(config_save_path, "w") as config_file:
        config_file.write(OmegaConf.to_yaml(config))


    adv = protect_image(
        config.method,
        pipe,
        src_image,
        mask_image_list,
        mask_image_combined,
        mask_radius_list,
        config
    )

    adv_image = tensor_to_pil_image(adv.detach().cpu())
    
    adv_image.save(os.path.join(output_img_folder, 'adv_image.png'))

if __name__ == "__main__":
    main()
