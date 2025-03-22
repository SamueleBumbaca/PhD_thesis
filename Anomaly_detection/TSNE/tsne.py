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
from models import get_model

def ensure_output_dir(model_name, dimension_reductor, dataset_name):
    """Create output directory structure if it doesn't exist"""
    output_dir = os.path.join("output", model_name, dimension_reductor, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def fix_random_seeds():
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

# Skips empty samples in a batch
def collate_skip_empty(batch):
    batch = [sample for sample in batch if sample] # check that sample is not None
    return torch.utils.data.dataloader.default_collate(batch)

def get_features(dataset, batch, num_images, dataset_class, model_name='resnet101'):
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Initialize the selected model
    model = get_model(model_name)
    model.eval()
    model.to(device)

    # read the dataset and initialize the data loader
    dataset = dataset_class(dataset, num_images)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, collate_fn=collate_skip_empty, shuffle=True)

    # we'll store the features as NumPy array of size num_images x feature_size
    features = None

    # we'll also store the image labels and paths to visualize them later
    labels = []
    image_paths = []

    for batch in tqdm(dataloader, desc=f'Running {model_name} inference'):
        images = batch['image'].to(device)
        labels += batch['label']
        image_paths += batch['image_path']

        with torch.no_grad():
            output = model(images)

        current_features = output.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features

    return features, labels, image_paths

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def scale_image(image, max_image_size):
    image_width, image_height = image.size

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    # Use the newer resampling filter names for compatibility with newer Pillow versions
    try:
        # For newer Pillow versions
        resampling_filter = Image.Resampling.LANCZOS
    except AttributeError:
        # Fall back to old filter name for older Pillow versions
        resampling_filter = Image.ANTIALIAS

    image = image.resize((image_width, image_height), resampling_filter)
    return image

def visualize_tsne_points_html(tx, ty, labels, colors_per_class, output_dir):
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
        title='t-SNE Visualization',
        labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2', 'label': 'Class'},
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
    html_path = os.path.join(output_dir, 'tsne_points.html')
    fig.write_html(html_path)
    print(f"TSNE points visualization saved to {html_path}")


def draw_rectangle_by_class(image, label, colors_per_class):
    image_width, image_height = image.size

    # get the color corresponding to image class
    color = tuple(colors_per_class[label])
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), (image_width - 1, image_height - 1)], outline=color, width=5)

    return image


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_width, image_height = image.size

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y


def visualize_tsne_images(tx, ty, images, labels, colors_per_class, plot_size=1000, max_image_size=100):
    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tsne_plot = Image.new('RGB', (plot_size, plot_size), (255, 255, 255))

    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image_path, label, x, y in tqdm(
            zip(images, labels, tx, ty),
            desc='Building the T-SNE plot',
            total=len(images)
    ):
        image = Image.open(image_path)

        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)

        # draw a rectangle with a color corresponding to the image class
        image = draw_rectangle_by_class(image, label, colors_per_class)

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot.paste(image, (tl_x, tl_y))

    plt.imshow(np.array(tsne_plot))
    plt.show()


def visualize_tsne_points(tx, ty, labels, colors_per_class):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.show()


def visualize_tsne(tsne, images, labels, colors_per_class, plot_size=1000, max_image_size=100):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels, colors_per_class)

    # visualize the plot: samples as images
    visualize_tsne_images(tx, ty, images, labels, colors_per_class, plot_size=plot_size, max_image_size=max_image_size)


def visualize_tsne_histogram(tsne_score, labels, colors_per_class):
    """
    Visualize the 1D TSNE scores as histograms per class label.
    
    Args:
        tsne_score: 1D TSNE projection (n_samples, 1)
        labels: List of class labels for each sample
        colors_per_class: Dictionary mapping class labels to colors
    """
    # Flatten the TSNE scores if needed
    tsne_flat = tsne_score.flatten()
    
    # Create a new figure
    plt.figure(figsize=(12, 8))
    
    # Get unique labels
    unique_labels = set(labels)
    
    # For each class, create a separate histogram
    for label in unique_labels:
        # Get indices of samples from this class
        indices = [i for i, l in enumerate(labels) if l == label]
        
        # Get TSNE scores for this class
        class_scores = np.take(tsne_flat, indices)
        
        # Convert the class color to matplotlib format: BGR -> RGB, divide by 255
        color = np.array(colors_per_class[label][::-1], dtype=np.float) / 255
        
        # Plot histogram for this class
        plt.hist(class_scores, bins=20, alpha=0.7, 
                 label=f"{label} (n={len(indices)})", 
                 color=color)
    
    plt.title("Distribution of 1D TSNE Scores by Class")
    plt.xlabel("TSNE Score")
    plt.ylabel("Frequency")
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

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


def visualize_reduction(reduction_result, images, labels, colors_per_class, output_dir, method_name, max_image_size=100, vis_imgs=True):
    """Generic visualization function for any dimensionality reduction method"""
    # extract x and y coordinates
    tx = reduction_result[:, 0]
    ty = reduction_result[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # Save interactive HTML visualizations
    visualize_points_html(tx, ty, labels, colors_per_class, output_dir, method_name)
    
    if vis_imgs:
        # Save static image visualization
        visualize_images_static(tx, ty, images, labels, colors_per_class, output_dir, method_name, max_image_size)


def run_dimensionality_reduction(features, method="tsne", dimensions=None):
    """Run the selected dimensionality reduction method with multiple dimensions"""
    if dimensions is None:
        # Default 2D for visualization and 1D for anomaly detection
        dimensions = [2, 1]
    
    results = {}
    
    for dim in dimensions:
        print(f"Running {method.upper()} with {dim} dimensions...")
        
        if method.lower() == "tsne":
            result = TSNE(n_components=dim, random_state=42).fit_transform(features)
        elif method.lower() == "umap":
            result = UMAP(n_components=dim, random_state=42).fit_transform(features)
        elif method.lower() == "pca":
            pca = PCA(n_components=dim, random_state=42)
            result = pca.fit_transform(features)
            explained_var = sum(pca.explained_variance_ratio_)
            print(f"PCA {dim}D explained variance: {explained_var:.2f}")
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        results[dim] = result
    
    return results

def analyze_anomalies_multiple(result_1d, labels, output_dir, confidence=0.95):
    """
    Run multiple anomaly detection algorithms and compare their performance
    """
    import numpy as np
    from scipy import stats
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.mixture import GaussianMixture
    
    # Reshape data for sklearn
    result_1d_flat = result_1d.flatten()
    X = result_1d_flat.reshape(-1, 1)
    
    # Separate healthy and unhealthy samples
    healthy_indices = np.array([i for i, label in enumerate(labels) if label == "healthy"])
    unhealthy_indices = np.array([i for i, label in enumerate(labels) if label != "healthy"])
    
    if len(healthy_indices) == 0:
        with open(os.path.join(output_dir, "anomaly_detection_results.txt"), 'w') as f:
            f.write("No healthy samples found in the dataset.\n")
        return None
    
    X_healthy = X[healthy_indices]
    
    # Dictionary to store results of all methods
    results = {}
    
    # 1. IQR with Confidence Interval (your original method)
    healthy_scores = result_1d_flat[healthy_indices]
    
    # Calculate IQR for healthy samples
    q1 = np.percentile(healthy_scores, 25)
    q3 = np.percentile(healthy_scores, 75)
    iqr = q3 - q1
    
    # Define bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Filter out outliers to get inliers
    healthy_inliers = healthy_scores[(healthy_scores >= lower_bound) & 
                                     (healthy_scores <= upper_bound)]
    
    # Calculate confidence interval for inliers
    mean = np.mean(healthy_inliers)
    std = np.std(healthy_inliers)
    z = stats.norm.ppf((1 + confidence) / 2)
    
    ci_lower = mean - z * std
    ci_upper = mean + z * std
    
    # Generate predictions (1 for anomaly, 0 for normal)
    predictions_iqr = np.zeros(len(result_1d_flat))
    predictions_iqr[(result_1d_flat < ci_lower) | (result_1d_flat > ci_upper)] = 1
    
    # Store metrics
    results['IQR+CI'] = calculate_metrics(predictions_iqr, healthy_indices, unhealthy_indices)
    
    # 2. Isolation Forest
    model_if = IsolationForest(contamination=0.1, random_state=42)
    model_if.fit(X_healthy)
    
    # Get predictions (-1 for anomalies, 1 for normal)
    raw_preds = model_if.predict(X)
    predictions_if = np.zeros(len(raw_preds))
    predictions_if[raw_preds == -1] = 1  # Convert to binary (1 for anomaly)
    
    results['IsolationForest'] = calculate_metrics(predictions_if, healthy_indices, unhealthy_indices)
    
    # 3. One-Class SVM
    try:
        model_ocsvm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
        model_ocsvm.fit(X_healthy)
        
        # Get predictions
        raw_preds = model_ocsvm.predict(X)
        predictions_ocsvm = np.zeros(len(raw_preds))
        predictions_ocsvm[raw_preds == -1] = 1
        
        results['OneClassSVM'] = calculate_metrics(predictions_ocsvm, healthy_indices, unhealthy_indices)
    except Exception as e:
        print(f"OCSVM error: {e}")
        results['OneClassSVM'] = {"error": str(e)}
    
    # 4. Local Outlier Factor
    try:
        model_lof = LocalOutlierFactor(n_neighbors=min(20, len(X_healthy)-1), novelty=True)
        model_lof.fit(X_healthy)
        
        # Get predictions
        raw_preds = model_lof.predict(X)
        predictions_lof = np.zeros(len(raw_preds))
        predictions_lof[raw_preds == -1] = 1
        
        results['LOF'] = calculate_metrics(predictions_lof, healthy_indices, unhealthy_indices)
    except Exception as e:
        print(f"LOF error: {e}")
        results['LOF'] = {"error": str(e)}
    
    # 5. Gaussian Mixture Model
    try:
        model_gmm = GaussianMixture(n_components=1, random_state=42)
        model_gmm.fit(X_healthy)
        
        # Get log probability density
        log_probs = model_gmm.score_samples(X)
        
        # Define threshold (lower 1% of healthy samples' probabilities)
        threshold = np.percentile(model_gmm.score_samples(X_healthy), 1)
        
        # Points with lower probability are anomalies
        predictions_gmm = np.zeros(len(log_probs))
        predictions_gmm[log_probs < threshold] = 1
        
        results['GMM'] = calculate_metrics(predictions_gmm, healthy_indices, unhealthy_indices)
    except Exception as e:
        print(f"GMM error: {e}")
        results['GMM'] = {"error": str(e)}
    
    # Write results to file
    with open(os.path.join(output_dir, "anomaly_detection_results.txt"), 'w') as f:
        f.write(f"Anomaly Detection Results Comparison\n")
        f.write(f"==================================\n\n")
        f.write(f"Dataset Statistics:\n")
        f.write(f"  - Total samples: {len(result_1d_flat)}\n")
        f.write(f"  - Healthy samples: {len(healthy_indices)}\n")
        f.write(f"  - Unhealthy samples: {len(unhealthy_indices)}\n\n")
        
        f.write(f"Method Comparison:\n")
        f.write(f"{'-'*70}\n")
        f.write(f"{'Method':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}\n")
        f.write(f"{'-'*70}\n")
        
        best_accuracy = 0
        best_method = None
        
        for method, metrics in results.items():
            if "error" in metrics:
                f.write(f"{method:<15} Error: {metrics['error']}\n")
                continue
                
            f.write(f"{method:<15} {metrics['accuracy']:.4f}     {metrics['precision']:.4f}     "
                    f"{metrics['recall']:.4f}     {metrics['f1']:.4f}     {metrics['auc']:.4f}\n")
            
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_method = method
        
        f.write(f"{'-'*70}\n\n")
        
        if best_method:
            f.write(f"Best performing method: {best_method} (Accuracy: {best_accuracy:.4f})\n\n")
        
        # Add method-specific details
        f.write("Method Details:\n")
        f.write("1. IQR+CI: Interquartile Range with Confidence Interval\n")
        f.write(f"   - IQR bounds: [{lower_bound:.4f}, {upper_bound:.4f}]\n")
        f.write(f"   - Confidence interval ({confidence*100}%): [{ci_lower:.4f}, {ci_upper:.4f}]\n\n")
        
    print(f"Anomaly detection analysis saved to {output_dir}/anomaly_detection_results.txt")
    
    # Return the best accuracy for the dataframe
    return results ##best_accuracy if best_method else None

def calculate_metrics(predictions, healthy_indices, unhealthy_indices):
    """Calculate classification metrics for anomaly detection"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Create true labels (0 for healthy, 1 for unhealthy)
    true_labels = np.zeros(len(predictions))
    true_labels[unhealthy_indices] = 1
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'recall': recall_score(true_labels, predictions, zero_division=0),
        'f1': f1_score(true_labels, predictions, zero_division=0),
        'auc': roc_auc_score(true_labels, predictions) if len(np.unique(true_labels)) > 1 else 0
    }
    
    return metrics

def analyze_dimension_impact(results_df, output_path):
    """Generate analysis of how dimension affects anomaly detection performance"""
    import plotly.express as px
    
    # Group by dimension and method, calculate mean metrics
    dim_analysis = results_df.groupby(['Dimensions', 'AnomalyMethod']).mean().reset_index()
    
    # Create visualization
    fig = px.line(
        dim_analysis, x='Dimensions', y='Accuracy', color='AnomalyMethod',
        title='Impact of Dimensionality on Anomaly Detection Performance',
        log_x=True,  # Logarithmic x-axis for dimensions
        labels={'Dimensions': 'Number of Dimensions (log scale)', 
                'Accuracy': 'Mean Accuracy'},
        markers=True
    )
    
    fig.update_layout(
        xaxis_title='Dimensions (log scale)',
        yaxis_title='Mean Accuracy',
        legend_title='Anomaly Detection Method',
        width=1000,
        height=600
    )
    
    # Save the figure
    fig.write_html(output_path)
    print(f"Dimension impact analysis saved to {output_path}")

def classify_by_clusterization(result_1d, labels, uni_labels, output_dir):
    """
    Cluster 1D reduction results and calculate classification metrics compared to original labels.
    
    Args:
        result_1d: 1D dimensionality reduction results (n_samples, 1)
        labels: Original labels for each data point
        uni_labels: Unique labels in the dataset
    
    Returns:
        Dictionary of results for different clustering methods
    """
    import numpy as np
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import cohen_kappa_score
    import os
    
    # Reshape data for sklearn
    result_1d_flat = result_1d.flatten()
    X = result_1d_flat.reshape(-1, 1)
    
    n_clusters = len(uni_labels)
    
    # Dictionary to store results
    results = {}
    
    # Convert string labels to numeric for easier processing
    label_to_idx = {label: i for i, label in enumerate(uni_labels)}
    numeric_labels = np.array([label_to_idx[label] for label in labels])
    
    # Function to map cluster assignments to original labels
    def map_clusters_to_labels(cluster_assignments):
        mapped_clusters = np.zeros_like(cluster_assignments)
        
        # For each cluster, find the most common original label
        for cluster_id in np.unique(cluster_assignments):
            cluster_indices = np.where(cluster_assignments == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
                
            # Get the most common original label in this cluster
            cluster_labels = numeric_labels[cluster_indices]
            unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
            most_common_label = unique_labels[np.argmax(label_counts)]
            
            # Map this cluster to the most common label
            mapped_clusters[cluster_indices] = most_common_label
        
        return mapped_clusters
    
    # Try different clustering algorithms
    
    # 1. K-Means
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_clusters = kmeans.fit_predict(X)
        
        # Map clusters to original labels
        mapped_kmeans = map_clusters_to_labels(kmeans_clusters)
        
        # Calculate metrics
        cohen_k = cohen_kappa_score(numeric_labels, mapped_kmeans)
        
        results['KMeans'] = {
            'cohen_k': cohen_k
        }
    except Exception as e:
        print(f"KMeans error: {e}")
        results['KMeans'] = {"error": str(e)}
    
    # 2. Hierarchical Clustering
    try:
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        hierarchical_clusters = hierarchical.fit_predict(X)
        
        # Map clusters to original labels
        mapped_hierarchical = map_clusters_to_labels(hierarchical_clusters)
        
        # Calculate metrics
        cohen_k = cohen_kappa_score(numeric_labels, mapped_hierarchical)
        
        results['Hierarchical'] = {
            'cohen_k': cohen_k
        }
    except Exception as e:
        print(f"Hierarchical clustering error: {e}")
        results['Hierarchical'] = {"error": str(e)}
    
    # 3. Gaussian Mixture Model
    try:
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(X)
        gmm_clusters = gmm.predict(X)
        
        # Map clusters to original labels
        mapped_gmm = map_clusters_to_labels(gmm_clusters)
        
        # Calculate metrics
        cohen_k = cohen_kappa_score(numeric_labels, mapped_gmm)
        
        results['GMM'] = {
            'cohen_k': cohen_k
        }
    except Exception as e:
        print(f"GMM error: {e}")
        results['GMM'] = {"error": str(e)}
    
    # 4. DBSCAN (with automatic epsilon estimation)
    try:
        # Estimate epsilon as the average distance to the 2nd nearest neighbor
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        epsilon = np.mean(distances[:, 1])
        
        dbscan = DBSCAN(eps=epsilon, min_samples=5)
        dbscan_clusters = dbscan.fit_predict(X)
        
        # Handle noise points (labeled as -1)
        if -1 in dbscan_clusters:
            # Assign noise points to the nearest cluster
            noise_indices = np.where(dbscan_clusters == -1)[0]
            non_noise_indices = np.where(dbscan_clusters != -1)[0]
            
            if len(non_noise_indices) > 0:
                nn_for_noise = NearestNeighbors(n_neighbors=1)
                nn_for_noise.fit(X[non_noise_indices])
                closest_non_noise = nn_for_noise.kneighbors(X[noise_indices], return_distance=False)
                
                for i, idx in enumerate(noise_indices):
                    non_noise_idx = non_noise_indices[closest_non_noise[i][0]]
                    dbscan_clusters[idx] = dbscan_clusters[non_noise_idx]
        
        # If DBSCAN didn't find enough clusters, skip it
        if len(np.unique(dbscan_clusters)) < 2:
            results['DBSCAN'] = {"error": "Not enough clusters found"}
        else:
            # Map clusters to original labels
            mapped_dbscan = map_clusters_to_labels(dbscan_clusters)
            
            # Calculate metrics
            cohen_k = cohen_kappa_score(numeric_labels, mapped_dbscan)
            
            results['DBSCAN'] = {
                'cohen_k': cohen_k
            }
    except Exception as e:
        print(f"DBSCAN error: {e}")
        results['DBSCAN'] = {"error": str(e)}
    
    # Write results to file if output directory is provided
    try:
        with open(os.path.join(output_dir, "clustering_results.txt"), 'w') as f:
            f.write(f"Clustering Results Comparison\n")
            f.write(f"===========================\n\n")
            f.write(f"Dataset Statistics:\n")
            f.write(f"  - Total samples: {len(result_1d_flat)}\n")
            f.write(f"  - Number of classes: {len(uni_labels)}\n")
            f.write(f"  - Classes: {', '.join(str(label) for label in uni_labels)}\n\n")
            
            f.write(f"Method Comparison:\n")
            f.write(f"{'-'*50}\n")
            f.write(f"{'Method':<15} {'Cohen Kappa':<15}\n")
            f.write(f"{'-'*50}\n")
            
            best_kappa = -1  # Cohen's Kappa ranges from -1 to 1
            best_method = None
            
            for method, metrics in results.items():
                if "error" in metrics:
                    f.write(f"{method:<15} Error: {metrics['error']}\n")
                    continue
                    
                f.write(f"{method:<15} {metrics['cohen_k']:.4f}\n")
                
                if metrics['cohen_k'] > best_kappa:
                    best_kappa = metrics['cohen_k']
                    best_method = method
            
            f.write(f"{'-'*50}\n\n")
            
            if best_method:
                f.write(f"Best performing method: {best_method} (Cohen's Kappa: {best_kappa:.4f})\n\n")
            
            # Add method-specific details
            f.write("Method Details:\n")
            f.write("1. KMeans: K-Means clustering with k=number of classes\n")
            f.write("2. Hierarchical: Agglomerative clustering with n_clusters=number of classes\n")
            f.write("3. GMM: Gaussian Mixture Model with n_components=number of classes\n")
            f.write("4. DBSCAN: Density-Based Spatial Clustering with automatic epsilon estimation\n")
            f.write(f"   - Estimated epsilon: {epsilon:.4f}\n")
            f.write(f"   - Min samples: 5\n")
        
        print(f"Clustering results saved to {output_dir}/clustering_results.txt")
    except Exception as e:
        print(f"Error writing clustering results: {e}")
    
    return results

def main():
    import numpy as np
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='data/raw-img')
    parser.add_argument('--dataset_type', type=str, default='animals', 
                       choices=['animals', 'plantvillage', 'plantpathology'])
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--num_images', type=int, default=500)
    parser.add_argument('--test_dimensions', action='store_true',
        help='Test multiple dimensions for reduction techniques')
    
    # Get all available model names from the models module
    import models
    try:
        # Try the new function if available
        all_model_names = models.get_available_models()
    except AttributeError:
        try:
            # Fallback to directly accessing the module-level variable
            all_model_names = sorted(models.ALL_MODELS.keys())
        except AttributeError:
            # Try to get models from custom and torchvision model dictionaries
            custom_models = getattr(models, 'CUSTOM_MODELS', {})
            torchvision_models = getattr(models, 'torchvision_models', {})
            all_model_names = sorted(list(custom_models.keys()) + list(torchvision_models.keys()))
    
    parser.add_argument('--model', type=str, default='resnet101',
                      choices=all_model_names + ['all'])
                      
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

    # Determine which models to run
    if args.model == 'all':
        models_to_run = all_model_names
        print(f"Running all {len(models_to_run)} models. This may take a while...")
    else:
        models_to_run = [args.model]
    
    # Determine which methods to run
    if args.reduction == 'all':
        methods = ['tsne', 'umap', 'pca']
    else:
        methods = [args.reduction]

    # Generate logarithmically spaced dimensions if testing multiple dimensions
    if args.test_dimensions:
        import numpy as np
        # Generate logarithmically spaced dimensions from 1 to 100
        dimensions = np.unique(np.logspace(0, 2, 10).astype(int))
        print(f"Testing dimensions: {dimensions}")
    else:
        dimensions = [2, 1]  # Default dimensions (2D for viz, 1D for anomaly)


    # Create DataFrame with new column for dimensions
    df = pd.DataFrame(columns=[
        'Dataset', 'Model', 'DimReduction', 'Dimensions', 
        'AnomalyMethod', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'
    ])

    df_cls = pd.DataFrame(columns=[
        'Dataset', 'Model', 'DimReduction', 'Dimensions',
        'ClusterizationMethod', 'CohenK'])

    # Run each model
    for model_name in models_to_run:
        print(f"\n===== Processing model: {model_name} =====")
        try:
            # Extract features
            features, labels, image_paths = get_features(
                dataset=args.path,
                batch=args.batch,
                num_images=args.num_images,
                dataset_class=Dataset,
                model_name=model_name
            )
            
            uni_labels = np.unique(labels)

            # Run each dimensionality reduction method
            for method in methods:
                # Create output directory
                output_dir = ensure_output_dir(model_name, method.upper(), args.dataset_type)
                
                # Run dimensionality reduction with all dimensions
                dim_results = run_dimensionality_reduction(features, method=method, 
                                                           dimensions=dimensions)
                
                # Always visualize with 2D and 1D for consistency
                if 2 in dim_results:
                    visualize_reduction(dim_results[2], image_paths, labels, 
                                       colors_per_class, output_dir, method.upper(), 
                                       vis_imgs=False)
                
                if 1 in dim_results:
                    visualize_histogram_html(dim_results[1], labels, colors_per_class, 
                                            output_dir, method.upper())
                
                # Run anomaly detection on each dimension result
                for dim, result in dim_results.items():
                    # if dim == 2:  # Skip 2D for anomaly detection
                    #     continue
                        
                    # For dimensions > 1, we need to convert to 1D for anomaly detection
                    if dim > 1:
                        # Use PCA to reduce to 1D for anomaly detection
                        from sklearn.decomposition import PCA
                        result_1d = PCA(n_components=1).fit_transform(result)
                    else:
                        result_1d = result
                    
                    # Create subdirectory for this dimension
                    dim_dir = os.path.join(output_dir, f"dim_{dim}")
                    os.makedirs(dim_dir, exist_ok=True)
                    
                    # Run anomaly detection
                    results_dict = analyze_anomalies_multiple(result_1d, labels, dim_dir)
                    class_results = classify_by_clusterization(result_1d,labels,uni_labels, dim_dir)

                    # Add results to dataframe
                    if results_dict:
                        for method_name, metrics in results_dict.items():
                            if "error" not in metrics:
                                df = pd.concat([df, pd.DataFrame([{
                                    'Dataset': args.dataset_type,
                                    'Model': model_name,
                                    'DimReduction': method,
                                    'Dimensions': dim,
                                    'AnomalyMethod': method_name,
                                    'Accuracy': metrics['accuracy'],
                                    'Precision': metrics['precision'],
                                    'Recall': metrics['recall'],
                                    'F1': metrics['f1'],
                                    'AUC': metrics['auc']
                                }])], ignore_index=True)
                    
                    if class_results:
                        for method_name, metrics in class_results.items():
                            if "error" not in metrics:
                                df_cls = pd.concat([df_cls, pd.DataFrame([{
                                    'Dataset': args.dataset_type,
                                    'Model': model_name,
                                    'DimReduction': method,
                                    'Dimensions': dim,
                                    'ClusterizationMethod': method_name,
                                    'CohenK': metrics['cohen_k']
                                }])], ignore_index=True)
                
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            continue
    
    # Save results
    df.to_csv('output/model_comparison_results.csv', index=False)

    df_cls.to_csv('output/clusterization_results.csv', index=False)
    
    # Generate dimension analysis plots
    if args.test_dimensions:
        analyze_dimension_impact(df, 'output/dimension_analysis.html')
    
    print(f"Model comparison results saved to output/model_comparison_results.csv")

if __name__ == '__main__':
    main()