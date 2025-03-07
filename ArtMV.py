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
    num_train_epochs: int = 1
    max_train_steps: int = 450  # New: limit total training steps
    gradient_accumulation_steps: int = 1
    
    # Optimizer parameters
    learning_rate: float = 1e-4
    lr_scheduler: str = "linear"
    lr_warmup_steps: int = 100  # Reduced from 700 since we have fewer steps
    
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
            self.source_images.extend(list(dir.glob("**/*.jpg")))  # Get jpg files
            self.source_images.extend(list(dir.glob("**/*.png")))  # Get png files
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
    monet_refs = []
    monet_refs.extend(list(Config.MONET_DIR.glob("**/*.jpg"))[:5])
    
    vangogh_refs = []
    vangogh_refs.extend(list(Config.VANGOGH_DIR.glob("**/*.png"))[:5])
    
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
    best_loss = float('inf')
    patience = 50  # Steps to wait before early stopping
    steps_without_improvement = 0
    
    for epoch in range(config.num_train_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_train_epochs}")
        for step, batch in enumerate(progress_bar):
            if step >= config.max_train_steps:
                print(f"\nReached max steps ({config.max_train_steps})")
                break
            
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
            
            # Early stopping check
            if loss.item() < best_loss:
                best_loss = loss.item()
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
            
            if steps_without_improvement >= patience:
                print(f"\nEarly stopping at step {step}")
                break
    
    print("Saving model...")
    save_path = Config.MODELS_DIR / "monet_vangogh_style"
    pipeline = save_lora_model(pipeline, save_path)
    print(f"Model saved to {save_path}")
    
    return pipeline

def generate_images(
    pipeline: StableDiffusionPipeline,
    source_images: List[Path],
    device: str
) -> List[Image.Image]:
    """Generate a single test image"""
    pipeline.to(device)
    pipeline.unet.eval()
    
    # Get the base unet from the LoRA model
    base_unet = pipeline.unet.get_base_model()
    
    # Switch to img2img pipeline for generation
    from diffusers import StableDiffusionImg2ImgPipeline
    img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        unet=base_unet,  # Use the base unet, not the LoRA wrapper
        torch_dtype=torch.float16,
        safety_checker=None,
        use_safetensors=True
    ).to(device)
    
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    generated_images = []
    for source_path in source_images:
        image = Image.open(source_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = img2img(
                prompt="A Van Gogh masterpiece of tulips with extremely pronounced, thick impasto brushstrokes. Each petal and stem defined by single, decisive strokes. Heavy paint application with clear, bold outlines. Rich blues and reds against golden background, each brushstroke standing out in relief",
                image=image,
                strength=0.45,  # Increased from 0.40
                guidance_scale=9.5,  # Up from 8.5 for more definition
                num_inference_steps=200,  # Reduced for bolder strokes
                negative_prompt="subtle, blended, smooth, soft, detailed, intricate, small strokes, pointillism, dots, fragmented, busy, noisy"
            ).images[0]
            
            output_path = Config.OUTPUT_DIR / f"{source_path.stem}_output.png"
            output.save(output_path)
            generated_images.append(output)
            wandb.log({"output": wandb.Image(output)})
    
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

def save_lora_model(pipeline, save_path):
    """Save LoRA weights properly"""
    pipeline.unet.save_pretrained(save_path)
    return pipeline

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
    test_images = []
    test_images.extend(list(Config.VANGOGH_DIR.glob("**/*.png"))[:1])  # Get first png file
    generated = generate_images(
        pipeline=pipeline,
        source_images=test_images,
        device=device
    )
    print("Done!")

if __name__ == "__main__":
    main()
