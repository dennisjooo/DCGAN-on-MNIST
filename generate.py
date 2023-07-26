# Importing the libraries
import torch
import model as m
import matplotlib.pyplot as plt


def generate(generator, device, num_images=1, coding_size=100, seed=True):

    # Setting the generator to evaluation mode
    generator.eval()

    # Setting the seed if needed
    if seed:
        g = torch.Generator(device=device).manual_seed(42)
        noise = torch.randn(num_images, coding_size, device=device, generator=g)

    else:
        noise = torch.randn(num_images, coding_size, device=device)

    # Getting the generated images
    generated_images = generator(noise)

    # Returning the generated images
    return generated_images


def plot_images(images, num_images=1, figsize=(1, 1), cmap='gray', save=False, save_path=None):
    # Plotting the images
    plt.figure(figsize=figsize)

    # Iterating over the images
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].cpu().detach().numpy().reshape(28, 28), cmap=cmap)
        plt.axis('off')

    # Tight layout
    plt.tight_layout()

    # Saving the plot if needed
    if save:
        plt.savefig(save_path)

    # Showing the plot
    plt.show()


if __name__ == "__main__":

    # Setting the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Getting the models
    generator = m.Generator().to(device)

    # Loading the generator
    generator.load_state_dict(torch.load('models//generator.pth', map_location=torch.device(device)))

    # Generating the images
    generated_images = generate(generator, device, num_images=5)

    # Plotting the images
    plot_images(generated_images, num_images=5, figsize=(5, 5), 
                cmap='gray', save=False, save_path=None)
        
    