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
    resolution: int = 512  # Standard SD resolution, can reduce to 384 if memory is tight
    
    # Training parameters
    train_batch_size: int = 2  # Reduced from 4 to be safer on GPU memory
    num_train_epochs: int = 50  # Reduced from 100 for faster initial testing
    gradient_accumulation_steps: int = 4  # Increased to compensate for smaller batch size
    
    # Optimizer parameters
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"  # Changed from "constant" for better convergence
    lr_warmup_steps: int = 100  # Added warmup steps
    
    # Performance parameters
    mixed_precision: str = "fp16"
    seed: int = 42
    
    # LoRA specific parameters
    lora_r: int = 16
    lora_alpha: int = 32
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

def collect_artwork(artists: List[str] = ["Claude Monet", "Vincent van Gogh"]) -> Tuple[int, int]:
    """Collect artwork for specified artists using WikiArt retriever"""
    wikiart_dir = setup_wikiart_retriever()
    artwork_dir = Config.DATA_DIR / "wikiart-saved"
    
    monet_count, vangogh_count = 0, 0
    
    for artist in artists:
        print(f"\nFetching artwork for {artist}...")
        result = subprocess.run([
            "python3", 
            str(wikiart_dir / "wikiart.py"),
            "--datadir", str(artwork_dir),
            "fetch",
            "--only", artist
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error fetching {artist}'s artwork:")
            print(result.stderr)
            continue
            
        # Move files to appropriate directories
        artist_dir = artwork_dir / artist.replace(" ", "_")
        if artist == "Claude Monet":
            for img in artist_dir.glob("*.jpg"):
                img.rename(Config.MONET_DIR / img.name)
                monet_count += 1
        elif artist == "Vincent van Gogh":
            for img in artist_dir.glob("*.jpg"):
                img.rename(Config.VANGOGH_DIR / img.name)
                vangogh_count += 1
                
    return monet_count, vangogh_count

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
    
    # Enable memory optimizations for GPU
    if device == "cuda":
        pipeline.enable_model_cpu_offload()
    
    # Enable LoRA training
    pipeline.enable_lora_training()
    
    return pipeline, accelerator

class ArtworkDataset(Dataset):
    """Dataset for artwork images"""
    def __init__(self, source_dir: Path, target_dir: Path, resolution: int = 512):
        self.source_images = list(source_dir.glob("*.jpg"))
        self.target_images = list(target_dir.glob("*.jpg"))
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
            'source_images': self.transform(source_img),
            'target_images': self.transform(target_img)
        }

def train_loop(
    config: TrainingConfig,
    pipeline: StableDiffusionPipeline,
    accelerator: Accelerator,
    device: str
):
    """Training loop for LoRA fine-tuning"""
    
    # Prepare dataset
    dataset = ArtworkDataset(Config.MONET_DIR, Config.VANGOGH_DIR, config.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Prepare optimizer
    optimizer = torch.optim.AdamW(
        pipeline.unet.parameters(),
        lr=config.learning_rate
    )
    
    # Prepare scheduler
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(dataloader) * config.num_train_epochs
    )
    
    # Prepare for training
    pipeline.train()
    
    # Accelerator prep
    pipeline, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        pipeline, optimizer, dataloader, lr_scheduler
    )
    
    # Training loop
    total_steps = len(dataloader) * config.num_train_epochs
    progress_bar = tqdm(range(total_steps), desc="Training")
    
    for epoch in range(config.num_train_epochs):
        for batch in dataloader:
            with accelerator.accumulate(pipeline):
                # Convert images to latent space
                latents = pipeline.vae.encode(
                    batch["source_images"].to(device)
                ).latent_dist.sample() * 0.18215
                
                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, pipeline.scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device
                )
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
                
                # Predict the noise
                noise_pred = pipeline.unet(noisy_latents, timesteps)["sample"]
                
                # Calculate loss
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                
                # Update weights
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            progress_bar.update(1)
            wandb.log({
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "epoch": epoch
            })
            
        # Save checkpoint at each epoch
        if epoch % 10 == 0:
            pipeline.save_pretrained(
                Config.MODELS_DIR / f"checkpoint-{epoch}",
                safe_serialization=True
            )
    
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
    
    # Setup model and accelerator
    pipeline, accelerator = setup_model(training_config, device)
    print(f"Model loaded and configured for {device}")
    
    if device == "cuda":
        print("GPU Memory allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
        print("GPU Memory cached:", torch.cuda.memory_reserved() / 1e9, "GB")
    
    # Data collection can run in background
    if Config.MONET_DIR.exists() and Config.VANGOGH_DIR.exists():
        print("\nUsing existing artwork...")
        monet_count = len(list(Config.MONET_DIR.glob("*.jpg")))
        vangogh_count = len(list(Config.VANGOGH_DIR.glob("*.jpg")))
        print(f"Found {monet_count} Monet paintings and {vangogh_count} Van Gogh paintings")
        
        # Start training if we have data
        if monet_count > 0 and vangogh_count > 0:
            print("\nStarting training loop...")
            pipeline = train_loop(training_config, pipeline, accelerator, device)
            print("\nTraining completed!")
            
            # Generate sample images
            print("\nGenerating sample images...")
            test_images = list(Config.MONET_DIR.glob("*.jpg"))[:5]  # Use first 5 Monet paintings
            generated_images = generate_images(
                pipeline=pipeline,
                source_images=test_images,
                device=device
            )
            print(f"\nGenerated {len(generated_images)} images in Van Gogh style")
            print(f"Images saved in: {Config.OUTPUT_DIR}")
        else:
            print("\nNo artwork found. Please run data collection first.")
    else:
        print("\nArtwork directories not found. Please run data collection first.")

if __name__ == "__main__":
    main()
