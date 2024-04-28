import torch
import torch.nn as nn

from model import *
from preprocess import *
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_channels = 64
n_epochs = 20
alpha = 0.02
learning_rate = 0.001
batch_size = 2

image_loader = get_images(batch_size)
image_size = (256, 256)
square_size = 100
binary_mask = get_mask(image_size, square_size)

inpaint = FreeFormImageInpaint().to(device)
optimizer = torch.optim.Adam(inpaint.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    inpaint.train()
    train_loss = 0
    loss = 0
    for batch_idx, (data, _) in enumerate(image_loader):
        data = data.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # reshape binary mask to add batch_size dimension
        binary_mask = binary_mask.unsqueeze(0).expand(batch_size, -1, -1)
        # model forward
        x_hat = inpaint(data, binary_mask)
        # compute the loss
        loss = inpaint.loss_function(x_hat, data, binary_mask, alpha)
        # model backward
        loss.backward()
        # update the model paramters
        optimizer.step()
        # update running training loss
        train_loss += loss
    train_loss = train_loss/len(image_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
