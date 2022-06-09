from config import Config
from dataset import SiameseNetworkDataset
from torch.utils.data import DataLoader
from functions import imshow 

import os
import torch
import torchvision
import torch.nn.functional as F 
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms

from model import SiameseNetwork

folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((224,224)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)

test_dataloader = DataLoader(siamese_dataset,num_workers=0,batch_size=1,shuffle=True)
dataiter = iter(test_dataloader)
x0,_,_ = next(dataiter)

net = SiameseNetwork()
net.load_state_dict(torch.load('model_state.pth'))

_, x1, label2 = next(dataiter)
concatenated = torch.cat((x0,x1),0)

output1,output2 = net(Variable(x0),Variable(x1))
euclidean_distance = F.pairwise_distance(output1, output2)
imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))