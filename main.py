# Importing the libraries
import torch
import train as t
import model as m
import os
import matplotlib.pyplot as plt


def load_or_new(generator, discriminator, device, num_epochs, 
                train_dataloader, optimizer_generator, optimizer_discriminator, 
                criterion, model_path, batch_size=128, 
                load=True, retrain=False, save=True):
    
    # Loading the models
    if load:
        # Loading in the gan if it exists
        if os.path.exists(model_path['Generator']) and os.path.exists(model_path['Discriminator']):
            # Loading the models
            generator.load_state_dict(torch.load(model_path['Generator'], map_location=torch.device(device)))
            discriminator.load_state_dict(torch.load(model_path['Discriminator'], map_location=torch.device(device)))
    
    # Re-training the model if it is needed
    if retrain:
        losses_g, losses_d = t.train(num_epochs, train_dataloader, 
                                     generator, discriminator, optimizer_generator, 
                                     optimizer_discriminator, device, criterion, batch_size=batch_size)
    else:
        losses_g = []
        losses_d = []
        
    # Saving the models and logs if needed
    if save:
        # Saving the models
        torch.save(generator.state_dict(), model_path['Generator'])
        torch.save(discriminator.state_dict(), model_path['Discriminator'])
    
    # Returning the losses
    return losses_g, losses_d


def plot_losses(losses_g, losses_d, num_epochs=10, figsize=(5, 5), save=False, save_path=None):
    # Plotting the losses
    plt.figure(figsize=figsize)
    plt.plot(range(num_epochs), losses_g, label='Generator')
    plt.plot(range(num_epochs), losses_d, label='Discriminator')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend()
    plt.tight_layout()
    
    # Saving the plot if needed
    if save:
        plt.savefig(save_path)
    
    # Showing the plot
    plt.show()


if __name__ == "__main__":

    # Setting a dict of model paths
    model_paths = {'Generator':'models//generator.pth',
                   'Discriminator': 'models//discriminator.pth'}
    
    # Setting the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setting the hyperparameters
    batch_size = 64
    num_epochs = 10
    lr_g = 0.0002
    lr_d = 0.0002
    betas =(0.5, 0.999)

    # Downloading the dataset
    dataset = t.get_data(train=True, download=True, transform=True)

    # Getting the dataloader
    train_dataloader = t.get_dataloader(dataset=dataset, batch_size=batch_size)

    # Getting the models
    generator = m.Generator().to(device)
    discriminator = m.Discriminator().to(device)

    # Getting the optimizers
    optimizer_generator = t.get_optimizer(generator, lr=lr_g, beta1=betas[0], beta2=betas[1])
    optimizer_discriminator = t.get_optimizer(discriminator, lr=lr_d, beta1=betas[0], beta2=betas[1])

    # Getting the loss function
    criterion = t.get_loss()

    # Loading in the gan if it exists
    losses_g, losses_d = load_or_new(generator, discriminator, device, num_epochs, 
                                     train_dataloader, optimizer_generator, optimizer_discriminator, 
                                     criterion, model_paths, batch_size=batch_size, 
                                     load=True, retrain=True, save=False)
    
    # Plotting the losses
    plot_losses(losses_g, losses_d, num_epochs=num_epochs, 
                figsize=(5, 5), save=False, save_path=None)