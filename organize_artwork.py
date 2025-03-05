from pathlib import Path
import shutil

def organize_artwork():
    # Setup paths
    data_dir = Path('/root/Wk6/data')
    wikiart_dir = data_dir / 'wikiart-saved/images'
    monet_source = wikiart_dir / 'claude-monet'
    vangogh_source = wikiart_dir / 'vincent-van-gogh'
    monet_dest = data_dir / 'monet'
    vangogh_dest = data_dir / 'vangogh'
    
    # Ensure destination directories exist
    monet_dest.mkdir(parents=True, exist_ok=True)
    vangogh_dest.mkdir(parents=True, exist_ok=True)
    
    # Move Monet paintings
    monet_count = 0
    if monet_source.exists():
        print(f"\nProcessing Monet paintings from {monet_source}")
        # Handle year directories
        for year_dir in monet_source.glob('*'):
            if year_dir.is_dir():
                print(f"Processing year: {year_dir.name}")
                for img in year_dir.glob('*.jpg'):
                    dest_file = monet_dest / img.name
                    print(f"Moving: {img.name}")
                    try:
                        shutil.move(str(img), str(dest_file))
                        monet_count += 1
                    except Exception as e:
                        print(f"Error moving {img}: {e}")
    
    print(f"\nMoved {monet_count} Monet paintings to {monet_dest}")

if __name__ == "__main__":
    print("Starting artwork organization...")
    organize_artwork()
    print("\nOrganization complete!") 