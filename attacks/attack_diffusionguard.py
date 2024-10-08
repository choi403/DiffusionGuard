import torch
from torch import Tensor
from typing import Any
from diffusers import StableDiffusionPipeline

def attack_pipeline(
    pipe: StableDiffusionPipeline,
    pixel_values: Tensor,
    masked_image: Tensor,
    mask: Tensor,
    prompt: str = '',
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    batch_size: int = 4,
    **kwargs: Any
) -> Tensor:
    '''
    Pipeline attack function for DiffusionGuard. Takes in all inputs
    and outputs the adversarial loss.
    
    Args:
    pipe (StableDiffusionPipeline): A diffusers pipeline object to protect the image against.
    pixel_values (Tensor): Source image to protect.
    masked_image (Tensor): Source image element-wise multiplied by the binary mask.
    mask (Tensor): Binary mask input for masked inpainting.
    guidance_scale (float = 7.5): Classifier-free guidance scale for text-to-image generation.
    prompt (str = ''): Editing prompt to be used for adversarial perturbation.
    height (int = 512): Height of the source image.
    width (int = 512): Width of the source image.
    num_inference_steps (int = 50): Number of inference steps to use for perturbation.
    batch_size (int = 4): Batch size for learning the perturbation.
    
    Returns:
    Tensor: The resulting adversarial loss of the source image plus perturbation.
    '''
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    text_embeddings = pipe.text_encoder(text_input_ids.to(pipe.device))[0]    
    text_embeddings = text_embeddings.detach()
    text_embeddings = torch.cat([text_embeddings] * batch_size, dim=0)

    pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps_all = pipe.scheduler.timesteps.to(pipe.device)
    timesteps = timesteps_all[0].long()

    num_channels_latents = pipe.vae.config.latent_channels
    noisy_model_input_shape = (batch_size, num_channels_latents, height // 8, width // 8)
    latents = torch.randn(noisy_model_input_shape, device=pipe.device, dtype=text_embeddings.dtype)
    latents = latents * pipe.scheduler.init_noise_sigma

    mask = torch.nn.functional.interpolate(mask, size=(height // 8, width // 8))
    mask = torch.cat([mask] * batch_size, dim=0)

    masked_image_latents = pipe.vae.encode(masked_image).latent_dist.sample()
    masked_image_latents = 0.18215 * masked_image_latents
    masked_image_latents = torch.cat([masked_image_latents] * batch_size, dim=0)

    latent_model_input = torch.cat([latents, mask, masked_image_latents], dim=1)
    noise_pred = pipe.unet(latent_model_input, timesteps, encoder_hidden_states=text_embeddings)[0]

    loss = -(noise_pred.norm(p=2) / batch_size)

    return loss
