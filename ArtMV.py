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
from transformers import CLIPVisionModel, CLIPImageProcessor

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
    train_batch_size: int = 4
    num_train_epochs: int = 1  # Test run
    gradient_accumulation_steps: int = 1
    
    # Optimizer parameters
    learning_rate: float = 5e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500  # Longer warmup
    
    # Performance parameters
    mixed_precision: str = "fp16"
    seed: int = 42
    
    # LoRA specific parameters
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1

def setup_model(config: TrainingConfig, device: str):
    print("Loading Stable Diffusion model...")
    
    # Load CLIP vision model
    print("Loading CLIP vision model...")
    image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Simple model loading without accelerator
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        use_safetensors=True
    ).to(device)
    
    # Add image encoder to pipeline
    pipeline.image_encoder = image_encoder
    pipeline.image_processor = image_processor
    
    # Minimal LoRA config
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "conv",
            "to_q",
            "to_k", 
            "to_v",
            "to_out.0"
        ],
        lora_dropout=0.1,
        bias="none"
    )
    
    pipeline.unet = get_peft_model(pipeline.unet, lora_config)
    return pipeline

class ArtworkDataset(Dataset):
    """Simplified dataset for artwork images"""
    def __init__(self, source_dir: List[Path], resolution: int = 512):
        self.source_images = []
        for dir in source_dir:
            self.source_images.extend(list(dir.glob("**/*.jpg")))
        self.transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
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
    batch_size = 4  # Define batch size
    pipeline.unet.train()
    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=config.learning_rate)
    
    # Basic dataset and dataloader
    dataset = ArtworkDataset([Config.MONET_DIR, Config.VANGOGH_DIR], config.resolution)
    dataloader = DataLoader(dataset, 
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Drop incomplete final batch
    )
    
    # Load both style reference images
    monet_refs = list(Config.MONET_DIR.glob("**/*.jpg"))[:5]
    vangogh_refs = list(Config.VANGOGH_DIR.glob("**/*.jpg"))[:5]
    
    # Load metadata
    monet_meta = {
        "artist": "Monet",
        "style": "Impressionist",
        "period": "19th century"
    }
    
    vangogh_meta = {
        "artist": "Van Gogh",
        "style": "Post-Impressionist",
        "period": "19th century"
    }
    
    # Combine metadata into embeddings
    meta_prompt = f"{monet_meta['style']} and {vangogh_meta['style']} painting"
    meta_input = pipeline.tokenizer(
        [meta_prompt],
        return_tensors="pt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length
    ).input_ids.to(device)
    
    meta_embeds = pipeline.text_encoder(meta_input)[0]
    
    # Get CLIP embeddings for style images
    with torch.no_grad():
        monet_embeds = get_image_embeddings(pipeline, monet_refs, device)
        vangogh_embeds = get_image_embeddings(pipeline, vangogh_refs, device)
        
        # Combine style embeddings
        style_embeds = torch.cat([monet_embeds, vangogh_embeds], dim=0)
        style_embeds = style_embeds.mean(dim=0, keepdim=True)  # Average embedding
        
        # Project CLIP embeddings to text encoder dimension
        projection = torch.nn.Linear(style_embeds.shape[-1], pipeline.text_encoder.config.hidden_size).to(device)
        style_embeds = projection(style_embeds)
        
        # Expand to sequence length and batch size
        style_embeds = style_embeds.unsqueeze(1).expand(-1, 77, -1)  # Add sequence dimension
        style_embeds = style_embeds.expand(batch_size, -1, -1)  # Expand to batch size
    
    progress_bar = tqdm(dataloader)
    for epoch in range(config.num_train_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_train_epochs}")
        for batch in progress_bar:
            images = batch.to(device)
            
            # Forward pass
            with torch.autocast(device_type=device):
                # Convert images to latent space
                latents = pipeline.vae.encode(images).latent_dist.sample() * 0.18215
                
                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (batch_size,), device=device)  # Match batch size
                noisy_latents = latents + noise
                
                # Get prediction
                pred = pipeline.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=style_embeds
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
    
    # Store original processor and temporarily replace
    original_processor = pipeline.image_processor
    pipeline.image_processor = None
    
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
                prompt="A vibrant post-impressionist painting with bold brushstrokes and swirling patterns in the style of Van Gogh",
                image=tensor,
                num_inference_steps=50,
                guidance_scale=12.0,
                noise_scale=0.5
            ).images[0]
            
            # Save and log
            output_path = Config.OUTPUT_DIR / f"{source_path.stem}_output.png"
            output.save(output_path)
            generated_images.append(output)
            wandb.log({"output": wandb.Image(output)})
    
    # Restore original processor
    pipeline.image_processor = original_processor
    return generated_images

def get_image_embeddings(pipeline, image_paths, device):
    """Get CLIP embeddings for a list of images"""
    embeddings = []
    for path in image_paths:
        image = Image.open(path).convert('RGB')
        inputs = pipeline.image_processor(images=image, return_tensors="pt").to(device)
        
        # Get CLIP image embeddings
        image_embeds = pipeline.image_encoder(**inputs).last_hidden_state.mean(dim=1)
        embeddings.append(image_embeds)
    
    return torch.cat(embeddings, dim=0)

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
    test_image = list(Config.VANGOGH_DIR.glob("**/*.jpg"))[0]
    generated = generate_images(
        pipeline=pipeline,
        source_images=[test_image],
        device=device
    )
    print("Done!")

if __name__ == "__main__":
    main()
