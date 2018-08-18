import torch
from torch import nn

DIM = 64
BATCH_SZIE = 50
OUTPUT_DIM = 784
NOISE_DIM = 128

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.linear1 = nn.Sequential(
			nn.Linear(128, 4*4*4*DIM),
			nn.ReLU(True)
		)
		self.deconv1 = nn.Sequential(
			nn.ConvTranspose2d(4*DIM, 2*DIM, kernel_size=5),
			nn.ReLU(True)
		)
		self.deconv2 = nn.Sequential(
			nn.ConvTranspose2d(2*DIM, DIM, kernel_size=5),
			nn.ReLU(True)
		)
		self.deconv3 =  nn.ConvTranspose2d(DIM, 1, kernel_size=8, stride=2)
		self.sigmoid = nn.Sigmoid()

	def forward(self, z):
		out = self.linear1(z)
		out = out.view(-1, 4*DIM, 4, 4)
		out = self.deconv1(out)
		out = out[:, :, :7, :7] #what is the purpose of this
		out = self.deconv2(out)
		out = self.deconv3(out)
		out = self.sigmoid(out)
		out = out.view(-1, OUTPUT_DIM)
		return out


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, DIM, kernel_size=5, stride=2, padding=2),
			nn.ReLU(True)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(DIM, 2*DIM, kernel_size=5, stride=2, padding=2),
			nn.ReLU(True)
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(2*DIM, 4*DIM, kernel_size=5, stride=2, padding=2),
			nn.ReLU(True))
		self.linear1 = nn.Linear(4*4*4*DIM, 1)

	def forward(self, x):
		out = x.view(-1, 1, 28, 28)
		out = self.conv1(out)
		out = self.conv2(out)
		out = self.conv3(out)
		out = out.view(-1, 4*4*4*DIM)
		out = self.linear1(out)
		out = out.view(-1)
		return out 

def generate_images(generator):
	noise = torch.randn(BATCH_SZIE, NOISE_DIM)
	images = generator(noise)
	images = images.view(BATCH_SZIE, 28, 28)

	return images
