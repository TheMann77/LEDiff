import os
import argparse
from pathlib import Path
import time
from natsort import natsorted
from tqdm import tqdm

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionITMPipeline
from scipy.optimize import least_squares
# import torchprofile  # 未使用可按需开启


def parse_args():
    parser = argparse.ArgumentParser(description="ITM HDR inference with LEDiff")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder with input LDR images")
    parser.add_argument("--output_hdr_path", type=str, required=True, help="Folder to save output HDR results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    return parser.parse_args()


def hdr_luminance_model(params, ldr_non_overexposed, hdr_non_overexposed):
    gamma, exp = params
    hdr_estimated = (ldr_non_overexposed ** gamma) * 2 ** exp
    return hdr_estimated - hdr_non_overexposed


def optimize_gamma_exp(LDR, HDR, mask, threshold):
    ldr_non_overexposed = LDR[mask == 1]
    hdr_non_overexposed = HDR[mask == 1]
    initial_params = [2.4, 0.0]
    bounds = ([2.4, -float("inf")], [2.6, float("inf")])
    result = least_squares(
        hdr_luminance_model,
        initial_params,
        bounds=bounds,
        args=(ldr_non_overexposed, hdr_non_overexposed),
    )
    return result.x


def apply_gamma_exp(LDR, gamma, exp):
    return (LDR ** gamma) * 2 ** exp


def blend_images(LDR, HDR, mask):
    blended_img = mask * LDR + (1 - mask) * HDR
    return blended_img


def generate_soft_mask(y, thr=0.05):
    msk = np.max(y, axis=2)
    msk = np.minimum(1.0, np.maximum(0.0, (msk - 1.0 + thr) / thr))
    msk = np.expand_dims(msk, axis=2)
    msk = np.tile(msk, [1, 1, 3])
    return msk


def process_and_save(LDR_path, HDR_path, output_path, threshold=250):
    LDR = cv2.imread(LDR_path).astype(np.float32) / 255.0
    HDR = cv2.imread(HDR_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR).astype(np.float32)
    print("max HDR is", np.max(HDR))

    overexposed_mask = generate_soft_mask(LDR)
    non_overexposed_mask = 1 - overexposed_mask

    optimal_gamma, optimal_exp = optimize_gamma_exp(LDR, HDR, non_overexposed_mask, threshold)
    LDR_adjusted = apply_gamma_exp(LDR, optimal_gamma, optimal_exp)
    blended_img = blend_images(LDR_adjusted, HDR, non_overexposed_mask)

    cv2.imwrite(output_path, blended_img.astype(np.float32))
    cv2.imwrite(output_path.replace(".hdr", "_mask.png"), non_overexposed_mask * 255.0)


def main():
    args = parse_args()

    model_path = args.model_path
    image_folder = args.image_folder
    output_hdr_path = args.output_hdr_path
    seed = args.seed

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionITMPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
    pipe.to(device)
    pipe.model_path = model_path
    pipe.set_progress_bar_config(disable=True)

    captions = [
        "A tall glass of water placed on a wooden floor, illuminated by intense sunlight creating sharp bright highlights and shadows.",
        "A rocky shoreline at sunset with calm waters, the sky glowing softly above the horizon.",
        "A scenic landscape with rocky terrain, scattered trees, and an overcast sky.",
        "A close-up of a red wooden wall with white trim under bright sunlight, part of a traditional building.",
        "A small golden owl figurine on a wooden floor, bathed in strong sunlight with dramatic reflections and shadows.",
    ]

    in_dir = Path(image_folder)
    image_files = natsorted([p for p in in_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    os.makedirs(output_hdr_path, exist_ok=True)

    print("Generating images:")
    for idx, img_path in enumerate(tqdm(image_files)):
        str_prompt = captions[idx % len(captions)]
        npy_save_name = str(Path(output_hdr_path) / f"img_{idx:03d}")

        result = pipe(prompt=str_prompt, img_name=str(img_path), npy_save_name=npy_save_name, seed=seed).images

        hdr = np.exp(result[1])
        hdr_bgr = cv2.cvtColor(hdr.astype(np.float32), cv2.COLOR_RGB2BGR)
        out_name = str(Path(output_hdr_path) / f"hdr_itm_{idx:03d}.hdr")
        cv2.imwrite(out_name, hdr_bgr)




if __name__ == "__main__":
    main()
