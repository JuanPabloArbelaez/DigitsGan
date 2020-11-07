import torch
from torch import nn 
from torchvision.utils import make_grid
import matplotlib.pyplot as plt



def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images,
    size per image. Plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def get_generator_block(input_dim, output_dim):
    '''
    Function for returning a block of the generator's neural network
    given imput and output parameters.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        putput_dim: the dimension of the output vector, a scalar
    Returns: 
        a generator neural network layer, with a linear transformation, 
        followerd by a batch normalization and then a relu activation
    '''

    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )


class Generator(nn.Module):
    '''
    Generatior Class
    values:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )

    def forward(self, noise):
        '''
        Function for compelting a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)

    def get_gen(self):
        '''
        Returns: 
            the sequential model
        '''
        return self.gen
        

def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Gicen the dimensions (n_samples, z_dims)
    creates a tensor of that shape filled with random numbers from the norm
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    return torch.randn(size=(n_samples, z_dim), device=device)


def get_discriminator_block(input_dim, output_dim):
    '''
    Discriminator Block
    Function for returning a neural network of the discriminator, give input and output dims
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network, with a linear transformation
        followd by a nn.LeakyReLu activation with a negative slope of 0.2
    '''
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
    )


class Discriminator(nn.Module):
    '''
    Discriminator class
    values:
        im_dim: the dimension of the images, fitted for the dataset
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image
        returns a 1-dimension tensor representing fake/real
        Parameters: 
            image: a flattened image with dimension (im_dim)
        '''
        return self.disc(image)

    def get_disc(self):
        '''
        Returns: 
            the sequential model
        '''
        return self.disc


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z_dimensionsional noise
        disc: the discriminator model, which returns a single dimensional prediction real/fake
        criterion: the loss function, which should be used to compare
            the discriminator's predictions to the ground truth reality of the images being
            (fake=0, real=1)
        real: a batch of real images
        num_images: the number of images the generator should produce,
            which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''
    noise = get_noise(num_images, z_dim, device)
    fake_images = gen(noise).detach()
    prediction_fake = disc(fake_images)
    prediction_real = disc(real)
    fake_loss = criterion(prediction_fake, torch.zeros(size=(num_images, 1), device=device))
    real_loss = criterion(prediction_real, torch.ones(size=(num_images, 1), device=device))
    disc_loss = (fake_loss + real_loss) / 2
    
    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image, given z-dimensional noise
        disc: the discriminator model, which returns a single dimensional vector: real/fake
        criterion: the loss function, which should be used to compare the discriminator's predictions
            to the ground truth reality.
            (fake=0, real=1)
        num_images: the number of images the generator should produce,
            which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    '''
    noise = get_noise(num_images, z_dim, device)
    fake_images = gen(noise)
    disc_pred = disc(fake_images)
    gen_loss = criterion(disc_pred, torch.ones(size=(num_images, 1), device=device))

    return gen_loss