import torch
from wGAN_mnist import Generator, Discriminator

# Test Generator 
BATCH_SIZE = 50
NOISE_DIM = 128
noise = torch.randn(BATCH_SIZE, NOISE_DIM)
generator = Generator()
imgs = generator.forward(noise)
print("images shape", imgs.shape)

discriminator = Discriminator()
imgs = torch.randn(BATCH_SIZE, 1, 28, 28)
labels = discriminator(imgs)
print('labels', labels.shape)