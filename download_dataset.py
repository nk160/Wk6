from datasets import load_dataset
from pathlib import Path

def download_dataset():
    print("Downloading dataset...")
    dataset = load_dataset("nk160/monet-vangogh-artworks")
    
    # Create directories if they don't exist
    Path("data/monet").mkdir(parents=True, exist_ok=True)
    Path("data/vangogh").mkdir(parents=True, exist_ok=True)
    
    print("Dataset info:", dataset)
    print("\nFirst few entries:", dataset['train'][:5])
    
if __name__ == "__main__":
    download_dataset() 