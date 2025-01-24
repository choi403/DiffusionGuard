## Introduction

This is the official repo of the paper ["DiffusionGuard: A Robust Defense Against Malicious Diffusion-based Image Editing"](https://arxiv.org/abs/2410.05694) (ICLR 2025)

## Contents

- [Introduction](#introduction)
- [Installation](#installation)
  - [Python environments](#python-environments)
- [Protecting images against inpainting](#protecting-images-against-inpainting)
- [Configuration](#configuration)

## Installation

### Python environments

```
conda create -n diffusionguard python=3.10
conda activate diffusionguard
pip install -r requirements.txt # We recommend using torch version 2.1.1 and CUDA version 12.2 for best compatibility.
```

## Protecting images against inpainting

You can protect images from diffusion-based image inpainting by generating adversarial perturbations using `main.py`. All images must be placed in the `assets` folder.

To run DiffusionGuard on image `keanu.png` for mask image `keanu_mask.png` in `assets` folder, please run the following code:

```
python main.py --config-name diffusionguard orig_image_name='keanu.png' mask_image_names='["keanu_mask.png"]'
```

In this example, orig_image_name refers to the original image you wish to protect, and mask_image_names contains the list of masks applied for protection.

If multiple mask images are given, the code will automatically merge them by taking the union of all masks.

### Configuration

- `method`: Protection method. Defaults to `diffusionguard`.

- `orig_image_name`: Filename of the image to be protected. Should be placed in `assets` folder.

- `mask_image_names`: List of filenames for the masks to be used. Should be placed in `assets` folder.

- `model`: Specifies the models used.

  - `inpainting`: Inpainting model. Defaults to `runwayml/stable-diffusion-inpainting`.

- `training`:

  - `size`: Image resolution of the image to be protected. Image should be square. Defaults to `512`.

  - `iters`: Number of PGD optimization iterations.

  - `grad_reps`: Number of reps to accumulate gradients.

  - `batch_size`: Batch size for PGD iteration.

  - `eps`: Linf threshold value for the adversarial noise. Defaults to 16/255.

  - `step_size`: PGD step size. Defaults to 1/255.

  - `num_inference_steps`: Number of inference steps for the inpainting pipeline.

  - `mask`: Mask-related settings.

    - `generation_method`: Mask generation method. Choose from `single`, `global`, and `contour_shrink`.

    - `contour_strength`: Shrink strength for `contour_shrink` method. Defaults to `1.0`.

    - `contour_iters`: Shrink iterations for `contour_shrink` method. Defaults to `15`.

    - `contour_smoothness`: Contour smoothing parameter for `contour_shrink` method. Defaults to `0.1`.
