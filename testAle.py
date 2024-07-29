import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

# Parameters
dataroot = 'C:/Users/fstef/Desktop/PROGETTO_PRATI/celeba/DFSmall'
attr_path = 'C:/Users/fstef/Desktop/PROGETTO_PRATI/celeba/attributeSmall.txt'
selected_attrs = ['Smiling', 'Young', 'Male', 'Wearing_Hat']  # Example attributes
batch_size = 128
image_size = 64
nz = 100
ngf = 64
ndf = 64
num_epochs = 5
lr = 0.0002
beta1 = 0.5
ngpu = 1
label_dim = len(selected_attrs)

# Define the CelebA Dataset
class CelebA(data.Dataset):
    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()
        self.num_images = len(self.train_dataset) if mode == 'train' else len(self.test_dataset)
        print(f"Initialized {mode} dataset with {self.num_images} samples.")
        self.check_dataset()

    def preprocess(self):
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = [values[self.attr2idx[attr_name]] == '1' for attr_name in self.selected_attrs]

            if (i+1) < 50:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def check_dataset(self):
        missing_files = []
        for dataset in [self.train_dataset, self.test_dataset]:
            for filename, _ in dataset:
                image_path = os.path.join(self.image_dir, filename)
                if not os.path.isfile(image_path):
                    missing_files.append(image_path)
        if missing_files:
            print(f"Missing files: {len(missing_files)}")
            for path in missing_files:
                print(path)

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image_path = os.path.join(self.image_dir, filename)
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            # Return a blank image and the label
            return torch.zeros((3, image_size, image_size)), torch.FloatTensor(label)
        image = Image.open(image_path)
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        return self.num_images

def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)

    print(f"Number of images in {mode} dataset: {len(dataset)}")  # Debugging line

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader

if __name__ == '__main__':
    # Create the dataset and dataloader using the new method
    dataloader = get_loader(dataroot, attr_path, selected_attrs, image_size=image_size, 
                            batch_size=batch_size, mode='train', num_workers=2)

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

    # Define the Generator and Discriminator models
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
                nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, noise, labels):
            z_out = self.noise_block(noise)
            l_out = self.label_block(labels)
            
            z_out = torch.nn.functional.interpolate(z_out, size=(4, 4))
            l_out = torch.nn.functional.interpolate(l_out, size=(4, 4))

            x = torch.cat([z_out, l_out], dim=1)
            return self.main(x)

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()

            self.img_block = nn.Sequential(
                nn.Conv2d(3, ndf // 2, 4, 2, 1, bias=False),
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

    # Create the Generator and Discriminator
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # Apply the weights_init function to randomly initialize all weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # Create the loss function and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop (For example)
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # Update Discriminator
            netD.zero_grad()
            real, labels = data
            batch_size = real.size(0)
            real = real.to(device)
            labels = torch.cat([labels] * (image_size // 64), dim=0).unsqueeze(2).unsqueeze(3).expand(-1, -1, image_size, image_size).to(device)
            label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            output = netD(real, labels).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise, labels)
            label.fill_(0)
            output = netD(fake.detach(), labels).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            optimizerD.step()

            # Update Generator
            netG.zero_grad()
            label.fill_(1)
            output = netD(fake, labels).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print(f'[{epoch+1}/{num_epochs}] '
                  f'[{i}/{len(dataloader)}] '
                  f'D_loss: {errD_real.item() + errD_fake.item()} '
                  f'G_loss: {errG.item()} '
                  f'D(x): {D_x} '
                  f'D(G(z)): {D_G_z1} / {D_G_z2}')

        # Save the models
        torch.save(netG.state_dict(), f'generator_{epoch+1}.pth')
        torch.save(netD.state_dict(), f'discriminator_{epoch+1}.pth')

        # Save some samples
        with torch.no_grad():
            fixed_noise = torch.randn(64, nz, 1, 1, device=device)
            fixed_labels = torch.zeros(64, label_dim, image_size, image_size, device=device)
            fake = netG(fixed_noise, fixed_labels)
            vutils.save_image(fake.detach(), f'images_{epoch+1}.png', normalize=True)
