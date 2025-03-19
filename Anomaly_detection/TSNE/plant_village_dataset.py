
from os import path, listdir
import torch
from torchvision import transforms
import random
import re

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Color palette for different plant diseases
colors_per_class = {
    # 'Apple_scab': [254, 202, 87],
    # 'Apple_black_rot': [255, 107, 107], 
    # 'Apple_cedar_apple_rust': [10, 189, 227],
    # 'Apple_healthy': [16, 172, 132],
    # 'Blueberry_healthy': [128, 80, 128],
    # 'Cherry_healthy': [87, 101, 116],
    # 'Cherry_powdery_mildew': [52, 31, 151],
    # 'Corn_cercospora_leaf_spot': [0, 0, 0],
    # 'Corn_common_rust': [100, 100, 255],
    # 'Corn_healthy': [255, 159, 243],
    # 'Corn_northern_leaf_blight': [255, 190, 11],
    # 'Grape_black_rot': [45, 152, 218],
    # 'Grape_esca': [214, 48, 49],
    # 'Grape_healthy': [85, 239, 196],
    # 'Grape_leaf_blight': [129, 236, 236],
    # 'Orange_haunglongbing': [250, 177, 160],
    # 'Peach_bacterial_spot': [116, 185, 255],
    # 'Peach_healthy': [162, 155, 254],
    # 'Pepper_bacterial_spot': [223, 230, 233],
    # 'Pepper_healthy': [75, 123, 236],
    # 'Potato_early_blight': [165, 94, 234],
    # 'Potato_healthy': [74, 105, 189],
    # 'Potato_late_blight': [144, 148, 151],
    # 'Raspberry_healthy': [254, 121, 104],
    # 'Soybean_healthy': [37, 204, 247],
    # 'Squash_powdery_mildew': [234, 181, 67],
    # 'Strawberry_healthy': [190, 46, 221],
    # 'Strawberry_leaf_scorch': [115, 53, 26],
    # 'Tomato_bacterial_spot': [47, 204, 113],
    # 'Tomato_early_blight': [232, 67, 147],
    # Add more classes as needed
}

class PlantVillageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, num_images=1000):
        if not path.exists(data_path):
            raise Exception(data_path + ' does not exist!')

        self.data = []
        self.classes = set()

        # Get all folders (classes) in the data path
        folders = listdir(data_path)
        for folder in folders:
            # Clean up the class name
            label = folder.replace('___', '_').replace('__', '_')
            self.classes.add(label)
            
            # Check if the color is defined for this class, if not assign a default
            if label not in colors_per_class:
                colors_per_class[label] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

            full_path = path.join(data_path, folder)
            if path.isdir(full_path):
                images = listdir(full_path)
                current_data = [(path.join(full_path, image), label) for image in images 
                               if image.lower().endswith(('.png', '.jpg', '.jpeg'))]
                self.data += current_data

        # Limit the number of images
        num_images = min(num_images, len(self.data))
        self.data = random.sample(self.data, num_images)

        # Use the same transforms as the original dataset
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