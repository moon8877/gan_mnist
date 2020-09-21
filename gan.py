import torch 
import torch.nn as nn
import numpy as np
from torchvision.transforms import transforms
from torchvision import datasets
from PIL import Image
from torch.utils.data import DataLoader

class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(784,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,input):
        return self.main(input)

class generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(128,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,784),
            nn.Tanh()
        )
    def forward(self,input):
        return self.main(input)

def d_loss_function(input,targets):
    return nn.BCELoss()(input,targets)

def g_loss_function(inputs):
    targets = torch.ones([inputs.shape[0],1])
    return nn.BCELoss()(inputs,targets)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        
    ]
)
G = generator()
D = discriminator()

g_optimizer = torch.optim.Adam(G.parameters(),lr=0.0002)
d_optimizer = torch.optim.Adam(D.parameters(),lr=0.0002)


train_set = datasets.MNIST('./mnist',train=True,transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
for epoch in range(200):
    for time , data in enumerate(train_loader):

        time = time+1
        print(data)
        real_input = data[0]
        print(data[0])
        while(True):
            i=1
        real_input = real_input.view(-1,784)
        real_output = D(real_input)
        real_label = torch.ones(real_input.shape[0], 1)
        noise = (torch.rand(real_input.shape[0],128)-0.5)/0.5
        fake_input = G(noise)
        fake_output = D(fake_input)
        fake_label = torch.zeros(fake_input.shape[0],1)
        output = torch.cat((real_output,fake_output),0)
        targets = torch.cat((real_label,fake_label),0)
        d_optimizer.zero_grad()
        d_loss = d_loss_function(output,targets)
        d_loss.backward()
        d_optimizer.step()

        noise = (torch.rand(real_input.shape[0],128)-0.5)/0.5
        fake_input = G(noise)
        fake_output = D(fake_input)
        g_loss = g_loss_function(fake_output)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        if time % 100 == 0 or time == len(train_loader):
            print('[{}/{}] D_loss: {:.3f}  G_loss: {:.3f}'.format( time, len(train_loader), d_loss.item(), g_loss.item()))
    for i in range(fake_input.shape[0]):
        image = ((fake_input[1]+1)/2).view(28,28)
        image = transforms.ToPILImage()(image)
        image.save(".\image3\image%d_%d.jpg" %(epoch,i))
       
  






