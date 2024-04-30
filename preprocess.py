
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms

# Custom transformation to convert single-channel images to RGB
class ConvertToRGB(object):
    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 2:  # Check if the image has only one channel
            img = img.convert('RGB')  # Convert to RGB
        return img


def collate_fn(batch):
   batch = list(filter(lambda x: x is not None, batch))
   return torch.utils.data.dataloader.default_collate(batch) 


def get_images(batch_size):
    transform = transforms.Compose([
        ConvertToRGB(), 
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.6 * len(cifar10))
    val_size = len(cifar10) - train_size
    train_set, val_set = torch.utils.data.random_split(cifar10, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


def get_mask(image_size, square_size):
    mask = np.zeros(image_size, dtype=np.uint8)
    start = (image_size[0] - square_size) // 2
    end = start + square_size
    mask[start:end, start:end] = 1
    mask = np.asarray(mask, np.float32)
    mask = torch.from_numpy(mask)
    return mask


def get_random_mask(image_size, square_size):
    mask = np.zeros(image_size, dtype=np.uint8)
    
    # Randomly select the starting position of the square
    start_x = np.random.randint(0, image_size[0] - square_size + 1)
    start_y = np.random.randint(0, image_size[1] - square_size + 1)
    
    # Calculate the end position based on the starting position and square size
    end_x = start_x + square_size
    end_y = start_y + square_size
    
    # Set the values within the square region of mask to 1
    mask[start_x:end_x, start_y:end_y] = 1
    
    # Convert mask to a NumPy array of floats and then to a PyTorch tensor
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