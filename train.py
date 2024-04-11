input_channels = 64
n_epochs = 20
alpha = 0.02
learning_rate = 0.001
optimizer = torch.optim.Adam(inpaint.parameters(), lr=learning_rate)
masks = None

from torchsummary import summary
inpaint = FreeFormImageInpaint(input_channels).to(device)
summary(inpaint,(1,256,256), masks)

for epoch in range(n_epochs):
    inpaint.train()
    train_loss = 0
    loss = 0
    for batch_idx, (data, labels) in enumerate(loader_train):
        #TODO
        # clear the gradients of all optimized variables
        # forward pass:
        # calculate the loss using the loss function defined above
        # backward pass: compute gradient of the loss with respect to model parameters
        # perform a single optimization step (parameter update)
        # update running training loss
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # reshape the image into a vector
        # model forward
        x_hat, mu, logvar = inpaint(data, masks)
        # compute the loss
        loss = inpaint.loss_function(x_hat, data, masks, alpha)
        # model backward
        loss.backward()
        # update the model paramters
        optimizer.step()

        train_loss += loss
    train_loss = train_loss/len(loader_train)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
