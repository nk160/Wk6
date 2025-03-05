import os
import torch
from pathlib import Path
import wandb
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from peft import LoraConfig, get_peft_model
from typing import List

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
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

def setup_model(config: TrainingConfig, device: str):
    print("Loading Stable Diffusion model...")
    
    # Simple model loading without accelerator
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        use_safetensors=True
    ).to(device)
    
    # Minimal LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["conv_in"],
        lora_dropout=0.1,
        bias="none"
    )
    
    pipeline.unet = get_peft_model(pipeline.unet, lora_config)
    return pipeline

class ArtworkDataset(Dataset):
    """Simplified dataset for artwork images"""
    def __init__(self, source_dir: Path, resolution: int = 512):
        self.source_images = list(source_dir.glob("*.jpg"))
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.source_images)
    
    def __getitem__(self, idx):
        # Just load and transform source image
        image = Image.open(self.source_images[idx]).convert('RGB')
        return self.transform(image)

def train_loop(config: TrainingConfig, pipeline: StableDiffusionPipeline, device: str):
    """Simplified training loop"""
    pipeline.unet.train()
    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=config.learning_rate)
    
    # Basic dataset and dataloader
    dataset = ArtworkDataset(Config.MONET_DIR, config.resolution)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Get style prompt embeddings instead of empty
    text_input = pipeline.tokenizer(
        ["A painting in the style of Van Gogh"] * 1,  # Style prompt
        return_tensors="pt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length
    ).input_ids.to(device)
    
    encoder_hidden_states = pipeline.text_encoder(text_input)[0]
    
    progress_bar = tqdm(dataloader)
    for batch in progress_bar:
        images = batch.to(device)
        
        # Forward pass
        with torch.autocast(device_type=device):
            # Convert images to latent space
            latents = pipeline.vae.encode(images).latent_dist.sample() * 0.18215
            
            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (1,), device=device)
            noisy_latents = latents + noise
            
            # Get prediction
            pred = pipeline.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            
            loss = F.mse_loss(pred, latents)
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        progress_bar.set_description(f"Loss: {loss.item():.4f}")
        wandb.log({"loss": loss.item()})
    
    return pipeline

def generate_images(
    pipeline: StableDiffusionPipeline,
    source_images: List[Path],
    device: str
) -> List[Image.Image]:
    """Generate a single test image"""
    pipeline.to(device)
    pipeline.unet.eval()
    
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    
    generated_images = []
    for source_path in source_images:
        # Load and transform image
        image = Image.open(source_path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate one image
        with torch.no_grad():
            output = pipeline(
                prompt="",  # Empty prompt for style transfer
                image=tensor,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]
            
            # Save and log
            output_path = Config.OUTPUT_DIR / f"{source_path.stem}_output.png"
            output.save(output_path)
            generated_images.append(output)
            wandb.log({"output": wandb.Image(output)})
    
    return generated_images

def main():
    """Simplified main execution"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Basic wandb init
    wandb.init(project="monet-to-vangogh")
    
    # Load model
    pipeline = setup_model(TrainingConfig(), device)
    print("Model loaded")
    
    # Train
    print("\nStarting training...")
    pipeline = train_loop(TrainingConfig(), pipeline, device)
    print("Training completed")
    
    # Generate one test image
    print("\nGenerating test image...")
    test_image = list(Config.VANGOGH_DIR.glob("*.jpg"))[0]
    generated = generate_images(
        pipeline=pipeline,
        source_images=[test_image],
        device=device
    )
    print("Done!")

if __name__ == "__main__":
    main()
