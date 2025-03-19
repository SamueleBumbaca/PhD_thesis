import torch
from torchvision import models
from torch.hub import load_state_dict_from_url


# Define the architecture by modifying resnet.
# Original code is here
# https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py
class ResNet101(models.ResNet):
    def __init__(self, num_classes=1000, pretrained=True, **kwargs):
        # Start with standard resnet101 defined here
        super().__init__(block=models.resnet.Bottleneck, layers=[3, 4, 23, 3], num_classes=num_classes, **kwargs)
        if pretrained:
            # Use the direct URL instead of model_urls dictionary
            url = "https://download.pytorch.org/models/resnet101-63fe2227.pth"
            # For PyTorch 1.6+, you can use:
            try:
                # Try the newer API first
                weights = models.ResNet101_Weights.IMAGENET1K_V1
                self.load_state_dict(models.resnet101(weights=weights).state_dict())
            except (AttributeError, ImportError):
                # Fall back to the older URL-based method
                state_dict = load_state_dict_from_url(url, progress=True)
                self.load_state_dict(state_dict)
                
    # Reimplementing forward pass.
    # Replacing the following code
    # https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py#L197-L213
    def _forward_impl(self, x):
        # Standard forward for resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Notice there is no forward pass through the original classifier.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x
