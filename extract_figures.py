#!/usr/bin/env python3
"""
Extract embedded PNG images from Jupyter notebook and save to figures/ directory.

Usage:
    python3 extract_figures.py
"""

import json
import base64
import os
from pathlib import Path


# Figure naming based on what appears in the README
FIGURE_NAMES = [
    "data_distribution.png",
    "scaffold_split_comparison.png",
    "training_curves.png",
    "roc_curves.png",
    "attention_comparison.png",
    "attention_known_inhibitors.png",
    "generated_candidates_distribution.png",
    "pareto_frontier.png",
    "docking_validation.png",
    "lead_structures.png",
    "synthetic_accessibility.png",
    "admet_profile.png",
]


def extract_images_from_notebook(notebook_path: str, output_dir: str):
    """
    Extract all embedded PNG images from a Jupyter notebook.
    
    Args:
        notebook_path: Path to the .ipynb file
        output_dir: Directory to save extracted PNG files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Extract images
    image_count = 0
    
    for cell_idx, cell in enumerate(notebook['cells']):
        if 'outputs' not in cell:
            continue
            
        for output_idx, output in enumerate(cell['outputs']):
            if 'data' not in output:
                continue
                
            if 'image/png' not in output['data']:
                continue
            
            # Get base64-encoded image data
            image_data = output['data']['image/png']
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            
            # Determine filename
            if image_count < len(FIGURE_NAMES):
                filename = FIGURE_NAMES[image_count]
            else:
                filename = f"figure_{image_count + 1:02d}.png"
            
            # Save to file
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'wb') as f:
                f.write(image_bytes)
            
            print(f"✓ Extracted: {filename} (cell {cell_idx}, output {output_idx})")
            image_count += 1
    
    print(f"\nTotal images extracted: {image_count}")
    return image_count


def main():
    """Main execution."""
    notebook_path = "notebooks/GNN_Antibiotic_Discovery_GyrB.ipynb"
    output_dir = "figures"
    
    print(f"Extracting images from: {notebook_path}")
    print(f"Output directory: {output_dir}/\n")
    
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return 1
    
    count = extract_images_from_notebook(notebook_path, output_dir)
    
    if count == 0:
        print("Warning: No images found in notebook")
        return 1
    
    print(f"\n✓ Successfully extracted {count} figures to {output_dir}/")
    return 0


if __name__ == "__main__":
    exit(main())
