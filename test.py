import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
#non cancellare 

# Set random seed for reproducibility
manualSeed = 999
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Parameters
dataroot = "C:/Users/fstef/Desktop/PROGETTO PRATI/celeba"
workers = 2
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 5
lr = 0.0002
beta1 = 0.5
ngpu = 1
label_dim = 40

if __name__ == '__main__':
    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("Using device:", device)

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()


    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Generator Code
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            
            self.noise_block = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False), 
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True)
            )
            
            self.label_block = nn.Sequential(
                nn.ConvTranspose2d(label_dim, ngf * 4, 4, 1, 0, bias=False), 
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True)
            )
            
            self.main = nn.Sequential(
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, noise, labels):
            z_out = self.noise_block(noise)
            l_out = self.label_block(labels)
            
            # Make sure the spatial dimensions match
            z_out = torch.nn.functional.interpolate(z_out, size=(4, 4))
            l_out = torch.nn.functional.interpolate(l_out, size=(4, 4))

            x = torch.cat([z_out, l_out], dim=1)
            return self.main(x)

    # Create the generator
    netG = Generator().to(device)

    # Apply the weights_init function to randomly initialize all weights
    netG.apply(weights_init)

    # Print the model
    print(netG)
    
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()

            self.img_block = nn.Sequential(
                nn.Conv2d(nc, ndf // 2, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )

            self.label_block = nn.Sequential(
                nn.Conv2d(label_dim, ndf // 2, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )

            self.main = nn.Sequential(
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, img, label):
            img_out = self.img_block(img)
            lab_out = self.label_block(label)
            x = torch.cat([img_out, lab_out], dim=1)
            x = self.main(x)
            return x.view(img.size(0), -1).mean(dim=1)

    # Create the Discriminator
    netD = Discriminator().to(device)
    netD.apply(weights_init)
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # label preprocessing
    onehot = torch.eye(label_dim).view(label_dim, label_dim, 1, 1).to(device)
    fill = torch.zeros([label_dim, label_dim, image_size, image_size]).to(device)
    for i in range(label_dim):
        fill[i, i, :, :] = 1

    # Function to generate test sample
    def generate_test(fixed_noise, onehot, G):
        G.eval()
        inference_res = None
        for l in range(label_dim):
            c = (torch.ones(8) * l).type(torch.LongTensor)
            c_onehot = onehot[c].to(device)
            out = G(fixed_noise, c_onehot)
            if inference_res is None:
                inference_res = out
            else:
                inference_res = torch.cat([inference_res, out], dim=0)
        G.train()
        return inference_res

    # Training Loop
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(dataloader, 0):
            b_size = imgs.size(0)
            real_cpu = imgs.to(device)
            
            real_label = torch.full((b_size,), 1., device=device)
            fake_label = torch.full((b_size,), 0., device=device)
            
            netD.zero_grad()
            output = netD(real_cpu, fill[labels][:b_size].to(device))
            errD_real = criterion(output, real_label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise, fill[labels][:b_size].to(device))
            output = netD(fake.detach(), fill[labels][:b_size].to(device))
            errD_fake = criterion(output, fake_label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            output = netD(fake, fill[labels][:b_size].to(device)).view(-1)
            errG = criterion(output, real_label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                    f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                    f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

            G_losses.append(errG.item())
            D_losses.append(errD.item())
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise, fill[labels][:64].to(device)).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        torch.save(netG.state_dict(), "./res/netG_conditional.pth")

        # Plot the training losses
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        # Visualize the generated images
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Generated Images")
        plt.imshow(np.transpose(img_list[-1], (1,2,0)))
        plt.show()

        # Create a grid of the final generated images and plot them
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()

        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Generated Images")
        plt.imshow(np.transpose(img_list[-1], (1,2,0)))
        plt.show()
