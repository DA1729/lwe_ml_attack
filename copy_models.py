#!/usr/bin/env python3
"""
Copy Models Script

This script copies your existing trained models from the original location
to the new project structure.
"""

import os
import shutil
from pathlib import Path

def copy_models():
    """Copy models from original location to new project structure"""
    
    # Source directory (original models)
    source_dir = Path("../lwe_chunked_models")
    
    # Destination directory (new project)
    dest_dir = Path("models")
    
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' not found!")
        print("Please ensure your original models are in the parent directory")
        return False
    
    print(f"Copying models from {source_dir} to {dest_dir}")
    
    # Create destination directory
    dest_dir.mkdir(exist_ok=True)
    
    copied_files = 0
    
    # Copy all .h5 model files
    for model_file in source_dir.glob("bit_model_*.h5"):
        dest_file = dest_dir / model_file.name
        shutil.copy2(model_file, dest_file)
        copied_files += 1
        if copied_files % 50 == 0:
            print(f"Copied {copied_files} models...")
    
    # Copy metadata and training data
    for file_name in ["metadata.json", "training_data.pkl", "training_progress.json"]:
        source_file = source_dir / file_name
        if source_file.exists():
            dest_file = dest_dir / file_name
            shutil.copy2(source_file, dest_file)
            print(f"Copied {file_name}")
    
    print(f"\n✓ Successfully copied {copied_files} model files")
    print(f"✓ Models are now ready in the {dest_dir} directory")
    print(f"\nYou can now run the demos:")
    print(f"  python demos/fast_demo.py")
    print(f"  python demos/simple_demo.py")
    print(f"  python demos/full_demo.py")
    
    return True

if __name__ == "__main__":
    copy_models()