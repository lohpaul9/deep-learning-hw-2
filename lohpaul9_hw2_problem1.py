import torch
import torchvision.models as models
import torchvision.transforms.functional as F
from PIL import Image
from pprint import pprint
import matplotlib.pyplot as plt
import math
import numpy as np
from skimage import data

"""
1a) Pecularities:
- BN is before RelU
- They used the affine parameters for BN, but no weights for Conv layers
- They use conv1x1 with stride 2 to downsample
- They also use the same stride for the first conv layer

1b) Torch.eval is used to set the model to evaluation mode (no dropout, batch norm, etc.)
It is used during inference or validation to ensure that the model is not in training mode.

Torch.train is used to set the model to training mode. This is used during training to ensure that
the model is using dropout, batch norm, etc. and is updating the weights.

1c) TODO: insert drawing

1d) Biases should not have weight decay because it may be the case that 
the data is not centered at zero. Encouraging the bias to be small acts against minimizing the
MLE loss and will increase bias error in the model. 

Additionally, biases do not contribute to the network's capacity to overfit to spurious noise, 
so they should not be regularized.

"""

def print_resnet18_params():
    # Load the pretrained ResNet18 model
    model = models.resnet18(pretrained=True)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Size: {param.size()}")

def categorize_resnet_18_params():
    # Load ResNet-18 model
    model = models.resnet18(pretrained=True)

    # Define dictionaries to store parameter groups
    batchnorm_params = []
    bias_params = []
    other_params = []

    # Iterate over the model's named parameters
    for name, _ in model.named_parameters():
        if 'bn' in name and ('weight' in name or 'bias' in name):
            # Group (i): BatchNorm affine transform parameters
            batchnorm_params.append(name)
        elif 'bias' in name:
            # Group (ii): Biases of Conv2d and Linear layers
            bias_params.append(name)
        else:
            # Group (iii): All other parameters (e.g., Conv2d and Linear weights)
            other_params.append(name)

    # Summary of results
    pprint(f"BatchNorm affine parameters: {batchnorm_params}")
    pprint(f"Bias parameters: {bias_params}")
    pprint(f"Other parameters: {other_params}")

def print_conv_details(layer, name):
    print(f"Layer: {name} ({layer.__class__.__name__})")
    print(f"  - In Channels: {layer.in_channels}")
    print(f"  - Out Channels: {layer.out_channels}")
    print(f"  - Kernel Size: {layer.kernel_size}")
    print(f"  - Stride: {layer.stride}")
    print(f"  - Padding: {layer.padding}")
    print(f"  - Dilation: {layer.dilation}")
    print(f"  - Groups: {layer.groups}")
    print(f"  - Bias: {layer.bias is not None}")
    print()

def print_resnet18_layers():
    # Load the ResNet-18 model
    model = models.resnet18(pretrained=True)

    # Iterate through all the modules in the model
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            print_conv_details(layer, name)
        else:
            print(f"Layer: {name} ({layer.__class__.__name__})")



# Load the astronaut image
image = data.astronaut()
image_pil = Image.fromarray(image)

# Functions for augmentations with hyperparameters inside
def apply_shear_x(img, magnitude=0.2):
    shear_x_angle = math.degrees(math.atan(magnitude))
    return F.affine(img, angle=0.0, translate=[0, 0], scale=1.0,
                    shear=[shear_x_angle, 0.0],
                    interpolation=F.InterpolationMode.BILINEAR,
                    fill=0)

def apply_shear_y(img, magnitude=0.2):
    shear_y_angle = math.degrees(math.atan(magnitude))
    return F.affine(img, angle=0.0, translate=[0, 0], scale=1.0,
                    shear=[0.0, shear_y_angle],
                    interpolation=F.InterpolationMode.BILINEAR,
                    fill=0)

def apply_translate_x(img, magnitude=20):
    return F.affine(img, angle=0.0, translate=[int(magnitude), 0], scale=1.0,
                    shear=[0.0, 0.0], interpolation=F.InterpolationMode.BILINEAR, fill=0)

def apply_translate_y(img, magnitude=20):
    return F.affine(img, angle=0.0, translate=[0, int(magnitude)], scale=1.0,
                    shear=[0.0, 0.0], interpolation=F.InterpolationMode.BILINEAR, fill=0)

def apply_rotate(img, angle=15):
    return F.rotate(img, angle, interpolation=F.InterpolationMode.BILINEAR, fill=0)

def apply_brightness(img, factor=1.5):
    return F.adjust_brightness(img, factor)

def apply_color(img, factor=1.5):
    return F.adjust_saturation(img, factor)

def apply_contrast(img, factor=1.5):
    return F.adjust_contrast(img, factor)

def apply_sharpness(img, factor=1.5):
    return F.adjust_sharpness(img, factor)

def apply_posterize(img, bits=4):
    bits = max(1, min(8, int(bits)))  # Ensure bits is between 1 and 8
    return F.posterize(img, bits)

def apply_solarize(img, threshold=128):
    threshold = max(0, min(255, int(threshold)))  # Ensure threshold is between 0 and 255
    return F.solarize(img, threshold)

def apply_equalize(img):
    return F.equalize(img)
# Function to display images in a grid
def show_images_in_grid(images, titles, grid_size=(3, 4), figsize=(12, 9)):
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
    axes = axes.flatten()

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(np.asarray(img))
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def show_all_augmentations():
    # Collect all augmented images and their titles
    images = [
        image_pil,
        apply_shear_x(image_pil, magnitude=0.3),
        apply_shear_y(image_pil, magnitude=0.3),
        apply_translate_x(image_pil, magnitude=40),
        apply_translate_y(image_pil, magnitude=40),
        apply_rotate(image_pil, angle=30),
        apply_brightness(image_pil, factor=2.0),
        apply_color(image_pil, factor=2.0),
        apply_contrast(image_pil, factor=2.0),
        apply_sharpness(image_pil, factor=2.0),
        apply_posterize(image_pil, bits=3),
        apply_solarize(image_pil, threshold=100),
        apply_equalize(image_pil),
    ]

    titles = [
        'Original Image', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY',
        'Rotate', 'Brightness', 'Color', 'Contrast', 'Sharpness',
        'Posterize', 'Solarize', 'Equalize'
    ]

    # Display all images in a grid
    show_images_in_grid(images, titles)

# Call the function to display all transformations
show_all_augmentations()
