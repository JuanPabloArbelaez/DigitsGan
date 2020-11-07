from tqdm.auto import tqdm
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from gan_generation import *



# Training Parameters
CRITERION = nn.BCEWithLogitsLoss()
N_EPOCHS = 200
Z_DIMS = 64
DISPLAY_STEP = 50
BATCH_SIZE = 128
LR = 0.00001
DEVICE = 'cuda'

# Load MNIST dataset as tensors
dataloader = DataLoader(
    MNIST('.', download=True, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE,
    shuffle=True)

# Create Generator & Discriminator
gen = Generator(Z_DIMS).to(DEVICE)
gen_opt = torch.optim.Adam(gen.parameters(), lr=LR)
disc = Discriminator().to(DEVICE)
disc_opt = torch.optim.Adam(disc.parameters(), lr=LR)


def run_neural_networks():
    cur_step= 0 
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    gen_loss = False
    error = False

    for epoch in range(N_EPOCHS):
        print('Epoch')

        # Dataloader returns the batches
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)

            # Flatten the batch of the real images from the dataset
            real = real.view(cur_batch_size, -1).to(DEVICE)

            ## Update discriminator ##
            # zero out the gradients before backpropagation
            disc_opt.zero_grad()

            # Calculate discriminator loss
            disc_loss = get_disc_loss(gen, disc, CRITERION, real, cur_batch_size, Z_DIMS, DEVICE)

            # Update gradients
            disc_loss.backward(retain_graph=True)

            # Update optimizer
            disc_opt.step()

            ## Update Generator ##
            # zero out the gradients before backpropagation
            gen_opt.zero_grad()

            # Calculate generator loss
            gen_loss = get_gen_loss(gen, disc, CRITERION, cur_batch_size, Z_DIMS, DEVICE)

            # Update Gradients
            gen_loss.backward(retain_graph=True)

            # Update optimizer
            gen_opt.step()


            # Keep track of average discriminator loss
            mean_discriminator_loss += disc_loss.item() / DISPLAY_STEP

            # Keep track of average generator loss
            mean_generator_loss += gen_loss.item() / DISPLAY_STEP


            ## Visualization code ##
            if (cur_step % DISPLAY_STEP == 0) and cur_step > 0:
                print(f'Epoch: {epoch}, Step: {cur_step}, Generator Loss: {mean_generator_loss}, Discriminator: {mean_discriminator_loss}')
                if epoch > 199:
                    noise = get_noise(cur_batch_size, Z_DIMS, DEVICE)
                    fake = gen(noise)
                    show_tensor_images(fake)
                    show_tensor_images(real)
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0
            
            cur_step += 1


if __name__ == '__main__':
    run_neural_networks()
