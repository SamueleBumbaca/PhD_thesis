from os import path
import pandas as pd
import torch
from torchvision import transforms
import random

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Color palette for different plant diseases
colors_per_class = {
    'healthy': [16, 172, 132],       # Green
    'multiple_diseases': [255, 107, 107],  # Red
    'rust': [254, 202, 87],          # Yellow
    'scab': [10, 189, 227],          # Blue
}

class PlantPathologyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, num_images=1000):
        """
        Args:
            data_path: Path to the dataset directory
            num_images: Maximum number of images to use
        """
        # Check if the data path exists
        if not path.exists(data_path):
            raise Exception(data_path + ' does not exist!')

        # Load the CSV file with pandas
        csv_file = path.join(data_path, 'train.csv')
        if not path.exists(csv_file):
            raise Exception(f'CSV file {csv_file} does not exist!')

        df = pd.read_csv(csv_file)
        
        # Define the image directory
        img_dir = path.join(data_path, 'images')
        if not path.exists(img_dir):
            raise Exception(f'Image directory {img_dir} does not exist!')

        # Create data samples
        self.data = []
        for _, row in df.iterrows():
            image_id = row['image_id']
            # Try different extensions if the exact format is unknown
            for ext in ['.jpg', '.jpeg', '.png']:
                image_path = path.join(img_dir, f"{image_id}{ext}")
                if path.exists(image_path):
                    break
            else:
                # Skip if image doesn't exist with any extension
                continue
                
            # Find the disease label (the column with value 1)
            label_columns = ['healthy', 'multiple_diseases', 'rust', 'scab']
            label = next((col for col in label_columns if row[col] == 1), None)
            
            if label:
                self.data.append((image_path, label))
        
        # Limit the number of images
        num_images = min(num_images, len(self.data))
        self.data = random.sample(self.data, num_images)

        # Use the same transforms as the other datasets
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]

        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None

        dict_data = {
            'image': image,
            'label': label,
            'image_path': image_path
        }
        return dict_data