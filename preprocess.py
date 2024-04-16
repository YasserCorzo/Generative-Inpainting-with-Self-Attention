
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms


def get_images(batch_size):
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
    ])

    cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)

    return data_loader


def get_mask(image_size, square_size):
    mask = np.zeros(image_size, dtype=np.uint8)
    start = (image_size[0] - square_size) // 2
    end = start + square_size
    mask[start:end, start:end] = 1
    mask = np.asarray(mask, np.float32)
    mask = torch.from_numpy(mask)
    return mask


def get_masked_images(image_loader, binary_mask):

    for images, labels in image_loader:
        print('shape of images pytorch:', images.shape)
        
        images_np = images.numpy().transpose((0, 2, 3, 1))
        print('shape of images numpy:', images_np.shape)

        masked_images = images_np.copy()
        
        plt.figure()

        for i in range(len(images)):
            masked_images[i][binary_mask == 1] = 1

            plt.subplot(2, 3, 3 * i + 1)
            plt.imshow(images_np[i])
            plt.title('original image')
            plt.axis('off')

            plt.subplot(2, 3, 3 * i + 2)
            plt.imshow(binary_mask, cmap='gray')
            plt.title('mask')
            plt.axis('off')
        
            plt.subplot(2, 3, 3 * (i + 1))
            plt.imshow(masked_images[i])
            plt.title('masked image')
            plt.axis('off')
        
        plt.show()
        
        break

    return masked_images


batch_size = 2
image_loader = get_images(batch_size)

image_size = (256, 256)
square_size = 100
binary_mask = get_mask(image_size, square_size)

get_masked_images(image_loader, binary_mask)

