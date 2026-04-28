# LEDiff: Latent Exposure Diffusion for HDR Generation (CVPR2025)

<p align="center">
  <a href="https://lediff.mpi-inf.mpg.de/">
    <img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page">
  </a>
  <a href="https://lediff.mpi-inf.mpg.de/resource/LEDiff_Latent_Exposure_Diffusion_for_HDR_Generation_Supp.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-red" alt="Paper">
  </a>
</p>

---

## 📖 <span style="background-color:#f0f0f0; padding:4px 8px; border-radius:6px;">Introduction</span>
LEDiff is a latent diffusion framework for high dynamic range (HDR) image generation.  
The method builds on the [Hugging Face diffusers](https://huggingface.co/docs/diffusers/index) library and adapts exposure-aware latent fusion for robust HDR synthesis.  
It achieves high fidelity in both shadow and highlight regions while remaining efficient for fine-tuning with limited HDR data.

---

## ⚙️ <span style="background-color:#f0f0f0; padding:4px 8px; border-radius:6px;">Usage for HDR Generation and Reconstruction</span>
For HDR generation and reconstruction examples, please go to: examples/text_to_image/

You can find scripts for:
- **HDR Generation**
- **Inverse Tone Mapping**


## Setup
```shell
pip install -e .
cd examples/text_to_image
pip install -r requirements.txt
pip install -r requirements_flax.txt
```