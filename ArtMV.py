import os
import sys
import torch
from pathlib import Path
import wandb
from PIL import Image
import numpy as np
from tqdm import tqdm
import subprocess
from typing import List, Tuple
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
import torch.nn.functional as F
from accelerate import Accelerator
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from peft import LoraConfig, get_peft_model
import torch.nn as nn
from datetime import datetime
import time

# Project configuration
class Config:
    PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = PROJECT_ROOT / "data"
    MONET_DIR = DATA_DIR / "monet"
    VANGOGH_DIR = DATA_DIR / "vangogh"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # Create necessary directories
    for dir_path in [DATA_DIR, MONET_DIR, VANGOGH_DIR, OUTPUT_DIR, MODELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model parameters
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    resolution: int = 512
    
    # Training parameters
    train_batch_size: int = 8
    num_train_epochs: int = 1  # Changed from 10 to 1 for testing
    gradient_accumulation_steps: int = 1
    
    # Optimizer parameters
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    
    # Performance parameters
    mixed_precision: str = "fp16"
    seed: int = 42
    
    # LoRA specific parameters
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1

def setup_environment():
    """Setup and verify the environment"""
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize W&B
    wandb.init(
        project="monet-to-vangogh",
        config={
            "architecture": "stable-diffusion-lora",
            "dataset": "wikiart",
            "source_artist": "Monet",
            "target_artist": "Van Gogh"
        }
    )
    
    return device

def setup_wikiart_retriever() -> Path:
    """Clone and setup WikiArt retriever if not already present"""
    wikiart_dir = Config.PROJECT_ROOT / "wikiart"
    if not wikiart_dir.exists():
        print("Cloning WikiArt retriever...")
        subprocess.run([
            "git", "clone", 
            "https://github.com/nk160/wikiart.git",
            str(wikiart_dir)
        ], check=True)
    return wikiart_dir

def move_artwork(images_dir: Path, dest_dir: Path, artist_pattern: str) -> int:
    """Move artwork files with detailed logging"""
    count = 0
    print(f"\nSearching for {artist_pattern} in {images_dir}")
    
    # Check multiple possible path patterns
    possible_paths = [
        images_dir / artist_pattern.lower().replace(" ", "-"),  # vincent-van-gogh
        images_dir / "vincent-van-gogh",                       # direct path
        images_dir                                             # root images dir
    ]
    
    for search_path in possible_paths:
        print(f"Checking path: {search_path}")
        if search_path.exists():
            print(f"Found path: {search_path}")
            # Look for jpg files recursively
            for img_file in search_path.rglob("*.jpg"):
                try:
                    dest_file = dest_dir / img_file.name
                    print(f"Moving: {img_file} -> {dest_file}")
                    img_file.rename(dest_file)
                    count += 1
                except Exception as e:
                    print(f"Error moving {img_file}: {e}")
    
    print(f"Found and moved {count} {artist_pattern} paintings")
    return count

def collect_artwork():
    """Check for existing artwork or collect if needed"""
    # Check if we already have Van Gogh paintings
    vangogh_count = len(list(Config.VANGOGH_DIR.glob("*.jpg")))
    print(f"\nFound {vangogh_count} existing Van Gogh paintings in {Config.VANGOGH_DIR}")
    
    # Only try to move files if we don't have any
    if vangogh_count == 0:
        wikiart_dir = setup_wikiart_retriever()
        artwork_dir = Config.DATA_DIR / "wikiart-saved"
        images_dir = artwork_dir / "images"
        if images_dir.exists():
            vangogh_count = move_artwork(images_dir, Config.VANGOGH_DIR, "vincent_van_gogh")
            print(f"Moved {vangogh_count} Van Gogh paintings to {Config.VANGOGH_DIR}")
    
    return vangogh_count

def setup_model(config: TrainingConfig, device: str):
    """Setup the Stable Diffusion model with LoRA configuration"""
    print("Loading Stable Diffusion model...")
    
    # Initialize accelerator with simpler configuration
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision
    )

    # Load the Stable Diffusion model with memory optimizations
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="scheduler"
        )
    )
    
    # Setup LoRA configuration
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=config.lora_dropout,
        bias="none",
    )
    
    # Apply LoRA to the UNet
    pipeline.unet = get_peft_model(pipeline.unet, lora_config)
    
    # Keep pipeline on CPU and let accelerator handle device placement
    pipeline.to("cpu")
    
    return pipeline, accelerator

class ArtworkDataset(Dataset):
    """Dataset for artwork images"""
    def __init__(self, source_dir: Path, target_dir: Path, resolution: int = 512, device_dtype=torch.float16):
        self.source_images = list(source_dir.glob("*.jpg"))
        self.target_images = list(target_dir.glob("*.jpg"))
        self.device_dtype = device_dtype
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def __len__(self):
        return min(len(self.source_images), len(self.target_images))
    
    def __getitem__(self, idx):
        source_img = Image.open(self.source_images[idx]).convert('RGB')
        target_img = Image.open(random.choice(self.target_images)).convert('RGB')
        
        return {
            'source_images': self.transform(source_img).to(self.device_dtype),
            'target_images': self.transform(target_img).to(self.device_dtype)
        }

def train_loop(config: TrainingConfig, pipeline: StableDiffusionPipeline, accelerator: Accelerator, device: str):
    """Training loop for LoRA fine-tuning"""
    print("\nPreparing for training...")
    
    # Set UNet to training mode
    pipeline.unet.train()
    
    # Determine dtype
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Prepare dataset with correct dtype
    dataset = ArtworkDataset(Config.MONET_DIR, Config.VANGOGH_DIR, config.resolution, device_dtype=dtype)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    
    # Get text embeddings for unconditional generation
    text_input = pipeline.tokenizer(
        [""] * config.train_batch_size, 
        padding="max_length", 
        max_length=pipeline.tokenizer.model_max_length, 
        return_tensors="pt"
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        pipeline.unet.parameters(),
        lr=config.learning_rate
    )
    
    # Get scheduler
    num_update_steps_per_epoch = len(dataloader)
    num_training_steps = config.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Let accelerator handle device placement
    pipeline.unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        pipeline.unet, optimizer, dataloader, lr_scheduler
    )
    
    # Move other components to device after accelerator prep
    pipeline.vae = pipeline.vae.to(device=device, dtype=dtype)
    pipeline.text_encoder = pipeline.text_encoder.to(device=device, dtype=dtype)
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(device))[0].to(dtype=dtype)
    
    # Training loop
    progress_bar = tqdm(range(len(dataloader)))
    for epoch in range(config.num_train_epochs):
        for batch in dataloader:
            with accelerator.accumulate(pipeline.unet):
                # Move batch to device
                source_images = batch["source_images"].to(device=device, dtype=dtype)
                
                # Convert images to latent space
                with torch.no_grad():  # Don't track VAE gradients
                    latents = pipeline.vae.encode(source_images).latent_dist.sample() * 0.18215
                
                # Add noise
                noise = torch.randn_like(latents, device=device)
                timesteps = torch.randint(
                    0, pipeline.scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                )
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
                
                # Predict the noise
                noise_pred = pipeline.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings
                )["sample"]
                
                # Calculate loss (ensure same device)
                loss = F.mse_loss(noise_pred.float(), noise.float())
                
                # Backward pass with retain_graph
                accelerator.backward(loss, retain_graph=True)
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)  # More efficient cleanup
                
                progress_bar.update(1)
                wandb.log({"loss": loss.item()})
    
    return pipeline

def generate_images(
    pipeline: StableDiffusionPipeline,
    source_images: List[Path],
    device: str,
    num_samples: int = 3,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50
) -> List[Image.Image]:
    """Generate images using the trained model"""
    pipeline.to(device)
    pipeline.eval()
    pipeline.enable_vae_slicing()  # Memory optimization
    
    generated_images = []
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    print("\nGenerating images...")
    for source_path in tqdm(source_images, desc="Processing images"):
        # Load and preprocess source image
        source_image = Image.open(source_path).convert('RGB')
        source_tensor = transform(source_image).unsqueeze(0).to(device)
        
        # Generate multiple samples for each source image
        for i in range(num_samples):
            with torch.no_grad():
                # Generate image
                image = pipeline(
                    image=source_tensor,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                ).images[0]
                
                # Save the generated image
                output_path = Config.OUTPUT_DIR / f"{source_path.stem}_vangogh_style_{i}.png"
                image.save(output_path)
                generated_images.append(image)
                
                # Log to W&B
                wandb.log({
                    "generated_images": [wandb.Image(
                        image,
                        caption=f"Source: {source_path.name}, Sample {i}"
                    )]
                })
    
    return generated_images

def main():
    """Main execution function"""
    device = setup_environment()
    print("Environment setup completed")
    
    # Initialize training config
    training_config = TrainingConfig()
    
    # Force data collection first
    print("\nStarting artwork collection...")
    vangogh_count = collect_artwork()
    print(f"\nCollected {vangogh_count} Van Gogh paintings")
    
    # Setup model and accelerator
    pipeline, accelerator = setup_model(training_config, device)
    print(f"Model loaded and configured for {device}")
    
    if device == "cuda":
        print("GPU Memory allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
        print("GPU Memory cached:", torch.cuda.memory_reserved() / 1e9, "GB")
    
    # Check if we have data
    if vangogh_count > 0:
        print("\nStarting training loop...")
        pipeline = train_loop(training_config, pipeline, accelerator, device)
        print("\nTraining completed!")
        
        # Generate sample images
        print("\nGenerating sample images...")
        test_images = list(Config.VANGOGH_DIR.glob("*.jpg"))[:5]  # Use first 5 Van Gogh paintings
        generated_images = generate_images(
            pipeline=pipeline,
            source_images=test_images,
            device=device
        )
        print(f"\nGenerated {len(generated_images)} images in Van Gogh style")
        print(f"Images saved in: {Config.OUTPUT_DIR}")
    else:
        print("\nError: No artwork found after collection attempt.")

if __name__ == "__main__":
    main()
