import argparse
from tqdm import tqdm
import os
import matplotlib
# Set the matplotlib backend to avoid Qt/wayland errors
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')  # Use non-GUI backend if no display available
else:
    matplotlib.use('TkAgg')  # Use TkAgg backend which usually works well

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
import random
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from Anomaly_detection.TSNE.test_all_models import get_model

# Rest of the imports and functions remain the same, until we need to modify the visualize functions

def visualize_points_html(tx, ty, labels, colors_per_class, output_dir, method_name):
    """Generic function to visualize dimensionality reduction results"""
    # Create a dataframe for plotly
    df = pd.DataFrame({
        'x': tx,
        'y': ty,
        'label': labels
    })
    
    # Create a custom color map matching our colors_per_class
    color_map = {}
    for label, color in colors_per_class.items():
        # Convert BGR to RGB and to hex format for plotly
        rgb = tuple(color[::-1])  # BGR to RGB
        hex_color = f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})'
        color_map[label] = hex_color
    
    # Create the scatter plot
    fig = px.scatter(
        df, x='x', y='y', color='label',
        color_discrete_map=color_map,
        title=f'{method_name} Visualization',
        labels={'x': f'{method_name} Component 1', 'y': f'{method_name} Component 2', 'label': 'Class'},
        hover_data=['label']
    )
    
    # Improve layout
    fig.update_layout(
        legend_title_text='Classes',
        plot_bgcolor='white',
        width=1000, 
        height=800
    )
    
    # Save the figure as HTML
    html_path = os.path.join(output_dir, f'{method_name.lower()}_points.html')
    fig.write_html(html_path)
    print(f"{method_name} points visualization saved to {html_path}")


def visualize_histogram_html(score, labels, colors_per_class, output_dir, method_name):
    """Generic function to visualize 1D dimensionality reduction histograms"""
    # Flatten the scores if needed
    flat_score = score.flatten()
    
    # Create a dataframe
    data = []
    for i, label in enumerate(labels):
        data.append({
            'score': flat_score[i],
            'label': label
        })
    
    df = pd.DataFrame(data)
    
    # Create color map
    color_map = {}
    for label, color in colors_per_class.items():
        # Convert BGR to RGB and to hex format for plotly
        rgb = tuple(color[::-1])  # BGR to RGB
        hex_color = f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})'
        color_map[label] = hex_color
    
    # Create histogram figure
    fig = go.Figure()
    
    # Add histograms for each class
    for label in set(labels):
        class_scores = df[df['label'] == label]['score']
        fig.add_trace(go.Histogram(
            x=class_scores,
            name=f"{label} (n={len(class_scores)})",
            marker_color=color_map[label],
            opacity=0.7,
            nbinsx=20
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Distribution of 1D {method_name} Scores by Class",
        xaxis_title=f"{method_name} Score",
        yaxis_title="Frequency",
        legend_title="Classes",
        barmode='overlay',
        plot_bgcolor='white',
        width=1200, 
        height=800
    )
    
    # Save the figure as HTML
    html_path = os.path.join(output_dir, f'{method_name.lower()}_histogram.html')
    fig.write_html(html_path)
    print(f"{method_name} histogram visualization saved to {html_path}")


def visualize_images_static(tx, ty, images, labels, colors_per_class, output_dir, method_name, max_image_size=100, plot_size=1000):
    """Generic function to visualize dimensionality reduction results as images"""
    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    plot = Image.new('RGB', (plot_size, plot_size), (255, 255, 255))

    # now we'll put a small copy of every image to its corresponding coordinate
    for image_path, label, x, y in tqdm(
            zip(images, labels, tx, ty),
            desc=f'Building the {method_name} plot',
            total=len(images)
    ):
        image = Image.open(image_path)

        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)

        # draw a rectangle with a color corresponding to the image class
        image = draw_rectangle_by_class(image, label, colors_per_class)

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

        # put the image to its coordinates using numpy subarray indices
        plot.paste(image, (tl_x, tl_y))

    # Save the image plot
    img_path = os.path.join(output_dir, f'{method_name.lower()}_images.png')
    plot.save(img_path)
    print(f"{method_name} images visualization saved to {img_path}")


def visualize_reduction(reduction_result, images, labels, colors_per_class, output_dir, method_name, max_image_size=100):
    """Generic visualization function for any dimensionality reduction method"""
    # extract x and y coordinates
    tx = reduction_result[:, 0]
    ty = reduction_result[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # Save interactive HTML visualizations
    visualize_points_html(tx, ty, labels, colors_per_class, output_dir, method_name)
    
    # Save static image visualization
    visualize_images_static(tx, ty, images, labels, colors_per_class, output_dir, method_name, max_image_size)


def run_dimensionality_reduction(features, method="tsne"):
    """Run the selected dimensionality reduction method"""
    print(f"Running {method.upper()} dimensionality reduction...")
    
    if method.lower() == "tsne":
        result_2d = TSNE(n_components=2, random_state=42).fit_transform(features)
        result_1d = TSNE(n_components=1, random_state=42).fit_transform(features)
    elif method.lower() == "umap":
        result_2d = UMAP(n_components=2, random_state=42).fit_transform(features)
        result_1d = UMAP(n_components=1, random_state=42).fit_transform(features)
    elif method.lower() == "pca":
        pca_2d = PCA(n_components=2, random_state=42)
        pca_1d = PCA(n_components=1, random_state=42)
        result_2d = pca_2d.fit_transform(features)
        result_1d = pca_1d.fit_transform(features)
        
        # Report explained variance for PCA
        print(f"PCA 2D explained variance: {sum(pca_2d.explained_variance_ratio_):.2f}")
        print(f"PCA 1D explained variance: {sum(pca_1d.explained_variance_ratio_):.2f}")
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
    return result_2d, result_1d


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='data/raw-img')
    parser.add_argument('--dataset_type', type=str, default='animals', 
                       choices=['animals', 'plantvillage', 'plantpathology'])
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--num_images', type=int, default=500)
    
    # Get all available model names from the models module
    import Anomaly_detection.TSNE.test_all_models as models_module
    all_model_names = models_module.get_model.__globals__['all_models'].keys()
    
    parser.add_argument('--model', type=str, default='resnet101',
                       choices=sorted(all_model_names))
                       
    # Add dimensionality reduction method selection
    parser.add_argument('--reduction', type=str, default='tsne',
                       choices=['tsne', 'umap', 'pca', 'all'],
                       help='Dimensionality reduction method to use')
                       
    args = parser.parse_args()

    fix_random_seeds()

    # Choose dataset class based on type parameter
    if args.dataset_type == 'plantpathology':
        from plant_pathology_dataset import PlantPathologyDataset as Dataset, colors_per_class
    elif args.dataset_type == 'plantvillage':
        from plant_village_dataset import PlantVillageDataset as Dataset, colors_per_class
    else:
        from animals_dataset import AnimalsDataset as Dataset, colors_per_class

    features, labels, image_paths = get_features(
        dataset=args.path,
        batch=args.batch,
        num_images=args.num_images,
        dataset_class=Dataset,
        model_name=args.model
    )

    # Determine which methods to run
    if args.reduction == 'all':
        methods = ['tsne', 'umap', 'pca']
    else:
        methods = [args.reduction]
    
    # Run each selected dimensionality reduction method
    for method in methods:
        # Create output directory for this method's results
        output_dir = ensure_output_dir(args.model, method.upper(), args.dataset_type)
        
        # Run dimensionality reduction
        result_2d, result_1d = run_dimensionality_reduction(features, method=method)
        
        # Visualize the results
        visualize_reduction(result_2d, image_paths, labels, colors_per_class, output_dir, method.upper())
        visualize_histogram_html(result_1d, labels, colors_per_class, output_dir, method.upper())


if __name__ == '__main__':
    main()