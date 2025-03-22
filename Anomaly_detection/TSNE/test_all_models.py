import subprocess
import os
from models import get_model

def main():
    # Set common parameters
    dataset_path = "data/plant-pathology-2020-fgvc7"
    dataset_type = "plantpathology"
    batch_size = 32  # Lower batch size to avoid memory issues
    num_images = 200  # Lower number of images for faster testing
    
    # Get all model names
    model_dict = get_model.__globals__['torchvision_models']
    model_names = list(model_dict.keys())
    
    # Add custom models
    model_names += ['resnet101', 'dinov2_vitb14', 'dinov2_vits14']
    
    # Create summary file
    summary_file = os.path.join("output", "model_comparison_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Testing {len(model_names)} models on {dataset_type} dataset with {num_images} images\n\n")
    
    # Test each model
    for i, model_name in enumerate(model_names):
        print(f"\n[{i+1}/{len(model_names)}] Testing model: {model_name}")
        
        try:
            # Run the tsne.py script with the current model
            cmd = [
                "python3", "tsne.py",
                "--path", dataset_path,
                "--dataset_type", dataset_type,
                "--batch", str(batch_size),
                "--num_images", str(num_images),
                "--model", model_name
            ]
            
            # Run the command and capture output
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                status = "SUCCESS"
            else:
                status = f"FAILED: {result.stderr.strip()}"
                
            # Log to summary file
            with open(summary_file, 'a') as f:
                f.write(f"Model: {model_name} - Status: {status}\n")
                
        except Exception as e:
            print(f"Error testing model {model_name}: {e}")
            with open(summary_file, 'a') as f:
                f.write(f"Model: {model_name} - Status: ERROR: {str(e)}\n")
    
    print(f"\nTesting complete. Summary saved to {summary_file}")

if __name__ == "__main__":
    main()