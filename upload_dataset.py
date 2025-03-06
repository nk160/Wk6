from datasets import Dataset
from pathlib import Path
import pandas as pd

def create_dataset():
    print("Creating dataset...")
    # Create image paths and metadata
    monet_files = list(Path("data/monet").glob("**/*.jpg"))
    vangogh_files = list(Path("data/vangogh").glob("**/*.jpg"))
    
    print(f"Found {len(monet_files)} Monet images and {len(vangogh_files)} Van Gogh images")
    
    data = {
        "image_path": [str(p) for p in (monet_files + vangogh_files)],
        "artist": ["monet"] * len(monet_files) + ["vangogh"] * len(vangogh_files)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    print("Uploading to Hugging Face Hub...")
    # Create Dataset and push
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(
        "nk160/monet-vangogh-artworks",
        private=True
    )
    print("Upload complete!")

if __name__ == "__main__":
    create_dataset() 