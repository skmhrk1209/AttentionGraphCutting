import torch
from torch import nn
import numpy as np
import functools
import cv2


@torch.no_grad()
def heatmap(image, colormap='COLORMAP_JET'):
    device = image.device
    image = image.permute(0, 2, 3, 1)
    image = image.cpu().detach().numpy()
    image = (image * ((1 << 8) - 1)).astype(np.uint8)
    image = np.stack(list(map(functools.partial(cv2.applyColorMap, colormap=getattr(cv2, colormap)), image)))
    image = image.astype(np.float32) / ((1 << 8) - 1)
    image = torch.tensor(image)
    image = image.permute(0, 3, 1, 2)
    image = image.to(device)
    return image


@torch.no_grad()
def blend(*images, weights=None):
    weights = weights or [1] * len(images)
    weights = [weight / sum(weights) for weight in weights]
    image = sum(image * weight for image, weight in zip(images, weights))
    return image


@torch.no_grad()
def normalize(image, mean, std):
    mean = image.new_tensor(mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-2)
    std = image.new_tensor(std).unsqueeze(0).unsqueeze(-1).unsqueeze(-2)
    image = (image - mean) / std
    return image


@torch.no_grad()
def unnormalize(image, mean, std):
    mean = image.new_tensor(mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-2)
    std = image.new_tensor(std).unsqueeze(0).unsqueeze(-1).unsqueeze(-2)
    image = image * std + mean
    return image


@torch.no_grad()
def linear_map(image, in_min, in_max, out_min, out_max):
    image = (image - in_min) / (in_max - in_min) * (out_max - out_min) + out_max
    return image
