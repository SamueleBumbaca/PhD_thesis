import torch
import torch.nn as nn
from torchvision import models
from torch.hub import load_state_dict_from_url
import torchvision.transforms as transforms

# Base feature extractor class for all torchvision models
class TorchvisionFeatureExtractor(nn.Module):
    def __init__(self, create_model_fn, weights='DEFAULT'):
        super().__init__()
        self.model = create_model_fn(weights=weights)
        
        # Remove classification head for feature extraction
        if hasattr(self.model, 'fc'):
            self.feature_dim = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif hasattr(self.model, 'classifier'):
            if isinstance(self.model.classifier, nn.Sequential):
                self.feature_dim = self.model.classifier[0].in_features
            else:
                self.feature_dim = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif hasattr(self.model, 'heads'):
            # ViT models
            self.feature_dim = self.model.hidden_dim
            self.model.heads = nn.Identity()
        elif hasattr(self.model, 'head'):
            # Some newer models
            self.feature_dim = self.model.head.in_features
            self.model.head = nn.Identity()
            
    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
        return features

# Original ResNet101 implementation - kept for backward compatibility
class ResNet101(models.ResNet):
    def __init__(self, num_classes=1000, pretrained=True, **kwargs):
        # Start with standard resnet101 defined here
        super().__init__(block=models.resnet.Bottleneck, layers=[3, 4, 23, 3], num_classes=num_classes, **kwargs)
        if pretrained:
            url = "https://download.pytorch.org/models/resnet101-63fe2227.pth"
            try:
                weights = models.ResNet101_Weights.IMAGENET1K_V1
                self.load_state_dict(models.resnet101(weights=weights).state_dict())
            except (AttributeError, ImportError):
                state_dict = load_state_dict_from_url(url, progress=True)
                self.load_state_dict(state_dict)
                
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

# DINOv2 models using torch.hub instead of Transformers
class DINOv2Base(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # Load model directly from torch.hub
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        except Exception as e:
            # Try loading from local cache if hub download fails
            cache_path = f"/home/samuelebumbaca/.cache/torch/hub/checkpoints/{model_name}.pth"
            print(f"Loading DINOv2 from local cache: {cache_path}")
            self.model = self._create_model_from_checkpoints(model_name, cache_path)
            
    def _create_model_from_checkpoints(self, model_name, checkpoint_path):
        """Create model based on architecture name and load weights"""
        if 'vitb14' in model_name:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=False)
        elif 'vitl14' in model_name:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)
        elif 'vits14' in model_name:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        # Load checkpoint
        state_dict = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(state_dict)
        return model
        
    def forward(self, x):
        # The input x is already normalized in the dataset class
        with torch.no_grad():
            # Get CLS token as feature representation
            features = self.model.forward_features(x)
            if isinstance(features, dict):
                # Some models return a dict, extract the CLS token
                features = features['x_norm_clstoken']
            else:
                # Otherwise, we have a tensor and take the CLS token (first position)
                features = features[:, 0]
        return features

# Specific DINOv2 model versions
class DINOv2ViTB14(DINOv2Base):
    def __init__(self):
        super().__init__('dinov2_vitb14_pretrain')

class DINOv2ViTL14(DINOv2Base):
    def __init__(self):
        super().__init__('dinov2_vitl14_pretrain')

class DINOv2ViTS14(DINOv2Base):
    def __init__(self):
        super().__init__('dinov2_vits14_pretrain')

# Factory function to get model by name
def get_model(model_name):
    # Original custom models
    custom_models = {
        'resnet101': ResNet101,
        'dinov2_vitb14': DINOv2ViTB14,
        'dinov2_vitl14': DINOv2ViTL14, 
        'dinov2_vits14': DINOv2ViTS14,
    }
    
    # PyTorch standard models with weights
    torchvision_models = {
        # DenseNet family
        'densenet121': lambda: TorchvisionFeatureExtractor(models.densenet121),
        'densenet161': lambda: TorchvisionFeatureExtractor(models.densenet161),
        'densenet169': lambda: TorchvisionFeatureExtractor(models.densenet169),
        'densenet201': lambda: TorchvisionFeatureExtractor(models.densenet201),
        
        # EfficientNet family
        'efficientnet_b0': lambda: TorchvisionFeatureExtractor(models.efficientnet_b0),
        'efficientnet_b1': lambda: TorchvisionFeatureExtractor(models.efficientnet_b1),
        'efficientnet_b2': lambda: TorchvisionFeatureExtractor(models.efficientnet_b2),
        'efficientnet_b3': lambda: TorchvisionFeatureExtractor(models.efficientnet_b3),
        'efficientnet_b4': lambda: TorchvisionFeatureExtractor(models.efficientnet_b4),
        'efficientnet_b5': lambda: TorchvisionFeatureExtractor(models.efficientnet_b5),
        'efficientnet_b6': lambda: TorchvisionFeatureExtractor(models.efficientnet_b6),
        'efficientnet_b7': lambda: TorchvisionFeatureExtractor(models.efficientnet_b7),
        
        # EfficientNetV2 family
        'efficientnet_v2_s': lambda: TorchvisionFeatureExtractor(models.efficientnet_v2_s),
        'efficientnet_v2_m': lambda: TorchvisionFeatureExtractor(models.efficientnet_v2_m),
        'efficientnet_v2_l': lambda: TorchvisionFeatureExtractor(models.efficientnet_v2_l),
        
        # GoogleNet & Inception
        'googlenet': lambda: TorchvisionFeatureExtractor(models.googlenet),
        'inception_v3': lambda: TorchvisionFeatureExtractor(models.inception_v3),
        
        # MaxVit
        'maxvit_t': lambda: TorchvisionFeatureExtractor(models.maxvit_t),
        
        # MNASNet
        'mnasnet0_5': lambda: TorchvisionFeatureExtractor(models.mnasnet0_5),
        'mnasnet1_0': lambda: TorchvisionFeatureExtractor(models.mnasnet1_0),
        
        # MobileNet family
        'mobilenet_v2': lambda: TorchvisionFeatureExtractor(models.mobilenet_v2),
        'mobilenet_v3_small': lambda: TorchvisionFeatureExtractor(models.mobilenet_v3_small),
        'mobilenet_v3_large': lambda: TorchvisionFeatureExtractor(models.mobilenet_v3_large),
        
        # RegNet family
        'regnet_y_400mf': lambda: TorchvisionFeatureExtractor(models.regnet_y_400mf),
        'regnet_y_800mf': lambda: TorchvisionFeatureExtractor(models.regnet_y_800mf),
        'regnet_y_1_6gf': lambda: TorchvisionFeatureExtractor(models.regnet_y_1_6gf),
        'regnet_y_3_2gf': lambda: TorchvisionFeatureExtractor(models.regnet_y_3_2gf),
        'regnet_y_8gf': lambda: TorchvisionFeatureExtractor(models.regnet_y_8gf),
        'regnet_y_16gf': lambda: TorchvisionFeatureExtractor(models.regnet_y_16gf),
        'regnet_y_32gf': lambda: TorchvisionFeatureExtractor(models.regnet_y_32gf),
        'regnet_x_400mf': lambda: TorchvisionFeatureExtractor(models.regnet_x_400mf),
        'regnet_x_800mf': lambda: TorchvisionFeatureExtractor(models.regnet_x_800mf),
        'regnet_x_1_6gf': lambda: TorchvisionFeatureExtractor(models.regnet_x_1_6gf),
        'regnet_x_3_2gf': lambda: TorchvisionFeatureExtractor(models.regnet_x_3_2gf),
        'regnet_x_8gf': lambda: TorchvisionFeatureExtractor(models.regnet_x_8gf),
        'regnet_x_16gf': lambda: TorchvisionFeatureExtractor(models.regnet_x_16gf),
        'regnet_x_32gf': lambda: TorchvisionFeatureExtractor(models.regnet_x_32gf),
        
        # ResNet family
        'resnet18': lambda: TorchvisionFeatureExtractor(models.resnet18),
        'resnet34': lambda: TorchvisionFeatureExtractor(models.resnet34),
        'resnet50': lambda: TorchvisionFeatureExtractor(models.resnet50),
        # 'resnet101' already in custom models
        'resnet152': lambda: TorchvisionFeatureExtractor(models.resnet152),
        
        # ResNeXt family
        'resnext50_32x4d': lambda: TorchvisionFeatureExtractor(models.resnext50_32x4d),
        'resnext101_32x8d': lambda: TorchvisionFeatureExtractor(models.resnext101_32x8d),
        'resnext101_64x4d': lambda: TorchvisionFeatureExtractor(models.resnext101_64x4d),
        
        # ShuffleNet family
        'shufflenet_v2_x0_5': lambda: TorchvisionFeatureExtractor(models.shufflenet_v2_x0_5),
        'shufflenet_v2_x1_0': lambda: TorchvisionFeatureExtractor(models.shufflenet_v2_x1_0),
        'shufflenet_v2_x1_5': lambda: TorchvisionFeatureExtractor(models.shufflenet_v2_x1_5),
        'shufflenet_v2_x2_0': lambda: TorchvisionFeatureExtractor(models.shufflenet_v2_x2_0),
        
        # SqueezeNet family
        'squeezenet1_0': lambda: TorchvisionFeatureExtractor(models.squeezenet1_0),
        'squeezenet1_1': lambda: TorchvisionFeatureExtractor(models.squeezenet1_1),
        
        # Swin Transformer family
        'swin_t': lambda: TorchvisionFeatureExtractor(models.swin_t),
        'swin_s': lambda: TorchvisionFeatureExtractor(models.swin_s),
        'swin_b': lambda: TorchvisionFeatureExtractor(models.swin_b),
        'swin_v2_t': lambda: TorchvisionFeatureExtractor(models.swin_v2_t),
        'swin_v2_s': lambda: TorchvisionFeatureExtractor(models.swin_v2_s),
        'swin_v2_b': lambda: TorchvisionFeatureExtractor(models.swin_v2_b),
        
        # VGG family
        'vgg11': lambda: TorchvisionFeatureExtractor(models.vgg11),
        'vgg11_bn': lambda: TorchvisionFeatureExtractor(models.vgg11_bn),
        'vgg13': lambda: TorchvisionFeatureExtractor(models.vgg13),
        'vgg13_bn': lambda: TorchvisionFeatureExtractor(models.vgg13_bn),
        'vgg16': lambda: TorchvisionFeatureExtractor(models.vgg16),
        'vgg16_bn': lambda: TorchvisionFeatureExtractor(models.vgg16_bn),
        'vgg19': lambda: TorchvisionFeatureExtractor(models.vgg19),
        'vgg19_bn': lambda: TorchvisionFeatureExtractor(models.vgg19_bn),
        
        # Vision Transformer family
        'vit_b_16': lambda: TorchvisionFeatureExtractor(models.vit_b_16),
        'vit_b_32': lambda: TorchvisionFeatureExtractor(models.vit_b_32),
        'vit_l_16': lambda: TorchvisionFeatureExtractor(models.vit_l_16),
        'vit_l_32': lambda: TorchvisionFeatureExtractor(models.vit_l_32),
        'vit_h_14': lambda: TorchvisionFeatureExtractor(models.vit_h_14),
        
        # Wide ResNet family
        'wide_resnet50_2': lambda: TorchvisionFeatureExtractor(models.wide_resnet50_2),
        'wide_resnet101_2': lambda: TorchvisionFeatureExtractor(models.wide_resnet101_2),
    }
    
    # Combine dictionaries, with custom models taking precedence
    all_models = {**torchvision_models, **custom_models}
    
    if model_name not in all_models:
        available_models = list(all_models.keys())
        raise ValueError(f"Model '{model_name}' not recognized. Available models: {available_models}")
    
    model_fn = all_models[model_name]
    return model_fn()