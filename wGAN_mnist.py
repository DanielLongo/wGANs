import torch
from torch import nn
from torch import optim
import torchvision.datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DIM = 64
BATCH_SIZE = 16
OUTPUT_DIM = 784
NOISE_DIM = 128
EPOCHS = 10
LR = 1e-4
torch.set_default_tensor_type('torch.cuda.FloatTensor')
transform = torchvision.transforms.ToTensor()
mnist_train = torchvision.datasets.MNIST('./MNIST_data', train=True, download=True, transform=transform)
TRAIN_LOADER = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)

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
	noise = torch.randn(BATCH_SIZE, NOISE_DIM)
	images = generator(noise)
	images = images.view(BATCH_SIZE, 28, 28)
	return images

def propagate(x):
	z = torch.randn(BATCH_SIZE, NOISE_DIM)
	images_fake = generator(z)
	D_labels_fake = discriminator(images_fake)
	D_labels_real = discriminator(x)
	return D_labels_real, D_labels_fake

def reset_grad():
    generator.zero_grad()
    discriminator.zero_grad()

generator = Generator()
discriminator = Discriminator()

G_solver = optim.RMSprop(generator.parameters(), lr=LR)
D_solver = optim.RMSprop(discriminator.parameters(), lr=LR)

old_hash = None
for epoch in range(EPOCHS):
	for i, (examples, _) in enumerate(TRAIN_LOADER):
		examples = examples.cuda()
		curr_hash = hash(examples[0])
		if curr_hash == old_hash:
			# raise ValueError('Got same hash between batches!!!')
			print("same hash", "batch #:", i, "epoch:", epoch)
		old_hash = curr_hash

		reset_grad()
		D_labels_real, D_labels_fake = propagate(examples)

		if (i % 5) != 0: #train discrminator more EXPLAIN
			D_loss = (torch.mean(D_labels_real) - torch.mean(D_labels_fake))
			D_loss.backward()
			D_solver.step()
			#weight clipping 
			for p in discriminator.parameters(): #explain weight clippings EXLAIN
				p.data.clamp_(-.01, .01)
			continue

		G_loss = torch.mean(D_labels_fake)
		G_loss.backward()
		G_solver.step()

		if i % 1000 == 0:
			if i != 0:
				print("Generator Loss:", G_loss.detach().cpu().numpy())
				print("Discriminator Loss:", D_loss.detach().cpu().numpy())
			fig = plt.figure(figsize=(4, 4))
			gs = gridspec.GridSpec(4, 4)
			gs.update(wspace=.05, hspace=.05)

			images = generator(torch.randn(16, NOISE_DIM)).data.cpu().numpy()
			print(images.shape)
			for img_num, sample in enumerate(images):
				ax = plt.subplot(gs[img_num])
				plt.axis('off')
				ax.set_xticklabels([])
				ax.set_yticklabels([])
				ax.set_aspect('equal')
				plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

			filename = "train-" + str(epoch) + "-" + str(i) 
			print("file logged")
			plt.savefig("./generated_images/" + filename, bbox_inches="tight" )
			plt.close(fig)

















# def calc_gradient_penalty(discriminator, real_data, fake_data):
# 	alpha = torch.rand(real_data.shape)
# 	interpolates = alpha * real_data + ((1 - alpha) * fake_data)
# 	disc_interpolates = discriminator(interpolates)

# 	grads = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.shape),
# 		create_graph=True, retain_graph=True, only_inputs=True)
