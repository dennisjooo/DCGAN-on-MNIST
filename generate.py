# Importing the libraries
import torch
import model as m
import matplotlib.pyplot as plt
import argparse


def generate(generator, device, num_images=1, coding_size=100, seed=True):

    # Setting the generator to evaluation mode
    generator.eval()

    # Setting the seed if needed
    if seed:
        # Seed for reproducibility
        g = torch.Generator(device=device).manual_seed(42)

        # Creating the noise
        noise = torch.normal(mean=0, std=1, size=(num_images, coding_size), device=device, generator=g)
    else:

        # Creating the noise randomly
        noise = torch.normal(mean=0, std=1, size=(num_images, coding_size), device=device)

    # Generating the images
    images = generator(noise)

    # Returning the generated images
    return images


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
    # Setting the parser
    parser = argparse.ArgumentParser(description='Generating images using the generator')
    parser.add_argument('--model_path', type=str, default='models//generator.pth', help='Path to the generator')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to generate')
    parser.add_argument('--coding_size', type=int, default=100, help='Coding size of the generator')
    parser.add_argument('--seed', type=bool, default=True, help='Whether to use a seed or not')
    parser.add_argument('--save', type=bool, default=False, help='Whether to save the plot or not')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the plot')

    # Parsing the arguments
    args = parser.parse_args()

    # Getting the arguments
    model_path = args.model_path
    num_images = args.num_images
    coding_size = args.coding_size
    seed = args.seed
    save = args.save
    save_path = args.save_path

    # Setting the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Getting the models
    generator = m.Generator().to(device)

    # Loading the generator
    generator.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    # Generating the images
    generated_images = generate(generator, device, num_images=num_images, coding_size=coding_size,
                                seed=seed)

    # Plotting the images
    plot_images(generated_images, num_images=num_images, figsize=(num_images, 1), 
                cmap='gray', save=save, save_path=save_path)
        
    