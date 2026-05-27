# Anime Face Latent Diffusion

This repository contains an experimental image generation pipeline for anime face synthesis.  
The project uses a custom Variational Autoencoder (VAE) to compress images into a latent space and a UNet-based diffusion model to generate new latent representations.

## Features

- Custom convolutional VAE architecture
- Latent-space diffusion training
- UNet denoiser with time embeddings, attention blocks and self-conditioning
- EMA model for more stable sampling
- LPIPS perceptual loss and Sobel edge loss for sharper VAE reconstructions
- Training on high-resolution anime portrait images
- Image sampling and visualization utilities

## Pipeline

1. Train the VAE on anime face images.
2. Encode images into latent vectors.
3. Normalize and cache latent representations.
4. Train a diffusion UNet to denoise latent samples.
5. Decode generated latents back into images using the VAE decoder.

## Tech Stack

- Python
- PyTorch
- Albumentations
- OpenCV
- LPIPS
- KaggleHub
- Matplotlib

## Goal

The main goal of this project is to explore how latent diffusion models work by building a simplified image generation system from scratch, without relying on ready-made Stable Diffusion pipelines.

<img width="1436" height="1028" alt="изображение" src="https://github.com/user-attachments/assets/a53efe97-48ac-4a46-b451-106ed166b95e" />
