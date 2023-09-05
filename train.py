# Importing PyTorch
import torch
from torchvision import transforms, datasets
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def get_data(train=True, download=True, transform=True):

    # Defining the transformations
    transform_mnist = transforms.Compose([
                           transforms.Resize(28),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,)),
                       ]) if transform else None

    # Downloading the MNIST dataset
    mnist_train = datasets.MNIST(root='./data', train=train, download=download, 
                                 transform=transform_mnist)
    
    return mnist_train


def get_dataloader(dataset, batch_size=128, shuffle=True):
    
    # Creating the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader


def get_optimizer(model, lr=0.0002, beta1=0.5, beta2=0.999):
        
        # Defining the optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
        
        return optimizer


def get_loss():
        
    # Defining the loss function
    loss = torch.nn.BCELoss()
    
    return loss


def train_discriminator(discriminator, optimizer, real_data, fake_data, criterion, device):

    # Setting the gradients to zero
    optimizer.zero_grad()

    # Getting the discriminator's predictions on the real data
    prediction_real = discriminator(real_data)

    # Getting the discriminator's predictions on the fake data
    prediction_fake = discriminator(fake_data)

    # Calculating the loss of real images
    loss_real = criterion(prediction_real, torch.ones_like(prediction_real, device=device) + 0.1 * torch.rand(prediction_real.shape, device=device))
    loss_real.backward()
    
    # Calculating the loss of fake images
    loss_fake = criterion(prediction_fake, torch.zeros_like(prediction_fake, device=device) - 0.1 * torch.rand(prediction_fake.shape, device=device))
    loss_fake.backward()
    
    # Summing up the losses
    loss = loss_real + loss_fake

    # Updating the weights
    optimizer.step()

    # Returning the loss
    return loss


def train_generator(discriminator, optimizer, fake_data, criterion, device):

    # Setting the gradients to zero
    optimizer.zero_grad()

    # Getting the discriminator's predictions on the fake data
    prediction_fake = discriminator(fake_data)

    # Calculating the loss
    loss = criterion(prediction_fake, torch.ones_like(prediction_fake, device=device))

    # Backpropagating the loss
    loss.backward()

    # Updating the weights
    optimizer.step()

    # Returning the loss
    return loss


# Defining the training loop
def train(num_epochs, train_dataloader, generator, discriminator, optimizer_generator, 
          optimizer_discriminator, device, criterion, batch_size=128, noise=None):

    # List to store the losses
    epoch_losses_discriminator = []
    epoch_losses_generator = []

    # Training the model
    for epoch in range(num_epochs):
        # List to store the losses
        losses_discriminator = []
        losses_generator = []

        # Iterating over the batches
        for i, (images, _) in enumerate(train_dataloader):

            # Moving the images to the device
            images = images.to(device)

            # Creating the noise
            noise = torch.randn(batch_size, 100, device=device)

            # Training the discriminator
            loss_discriminator = train_discriminator(discriminator, optimizer_discriminator, images, 
                                                     generator(noise), criterion, device)

            # Training the generator
            loss_generator = train_generator(discriminator, optimizer_generator, 
                                             generator(noise), criterion, device)

            # Appending the losses
            losses_discriminator.append(loss_discriminator.item())
            losses_generator.append(loss_generator.item())
            
            # Printing the losses every 250 steps
            if i % 250 == 0:
                print('Epoch: %d/%d, Step: %d/%d, Loss D: %.4f, Loss G: %.4f' % (epoch + 1, num_epochs, i, len(train_dataloader), losses_discriminator[-1], losses_generator[-1]))

        # Appending the epoch losses
        epoch_losses_discriminator.append(np.mean(losses_discriminator))
        epoch_losses_generator.append(np.mean(losses_generator))

        # If noise is not none, generate images
        if noise is not None:
            # Generating the images
            generated_images = generator(noise).cpu().detach().numpy()

            # Plotting the images
            fig, ax = plt.subplots(4, 4, figsize=(5, 5))
            for i in range(4):
                for j in range(4):
                    ax[i, j].imshow(generated_images[i * 4 + j].reshape(28, 28), cmap='gray')
                    ax[i, j].axis('off')
            plt.tight_layout()
            plt.savefig('images//prog//epoch_' + str(epoch + 1) + '.png')
        
        # Printing the losses
        print('Epoch: %d/%d, Loss D: %.4f, Loss G: %.4f' % (epoch + 1, num_epochs, epoch_losses_discriminator[-1], epoch_losses_generator[-1]))

    return epoch_losses_discriminator, epoch_losses_generator

