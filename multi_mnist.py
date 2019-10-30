import torch
from torchvision import datasets
from torchvision import transforms
import numpy as np
import random
import json
import cv2
import os


def main(dataset, dirname, num_samples=4, num_trials=100, input_size=28, output_size=128):
    meta = {}
    for i in range(len(dataset)):
        images = []
        labels = []
        boxes = []
        for _ in range(random.choice(range(1, num_samples + 1))):
            for _ in range(num_trials):
                y1 = random.randint(0, output_size - input_size)
                x1 = random.randint(0, output_size - input_size)
                y2 = y1 + input_size
                x2 = x1 + input_size
                if not any((box[0] <= y1 < box[2] or box[0] <= y2 < box[2]) and
                           (box[1] <= x1 < box[3] or box[1] <= x2 < box[3]) for box in boxes):
                    boxes.append((y1, x1, y2, x2))
                    image, label = random.choice(dataset)
                    images.append(image)
                    labels.append(label)
                    break
        boxes, images, labels = zip(*sorted(zip(boxes, images, labels)))
        background = torch.zeros(torch.stack(images).size(1), output_size, output_size)
        for image, box in zip(images, boxes):
            background[..., box[0]:box[2], box[1]:box[3]] = image
        meta[f'{i}.png'] = dict(boxes=boxes, labels=labels)
        cv2.imwrite(os.path.join(dirname, f'{i}.png', (background.permute(1, 2, 0).numpy() * 255).astype(np.uint8)))
    with open(os.path.join(dirname, 'meta.json'), 'w') as file:
        json.dump(meta, file)


if __name__ == '__main__':

    main(
        dataset=datasets.MNIST(
            root='mnist',
            train=True,
            transform=transforms.ToTensor(),
            download=True
        ),
        dirname=os.path.join('multi_mnist', 'train'),
        num_samples=4,
        input_size=28,
        output_size=128
    )

    main(
        dataset=datasets.MNIST(
            root='mnist',
            train=False,
            transform=transforms.ToTensor(),
            download=True
        ),
        dirname=os.path.join('multi_mnist', 'test'),
        num_samples=4,
        input_size=28,
        output_size=128
    )
