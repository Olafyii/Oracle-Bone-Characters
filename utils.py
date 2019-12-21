from PIL import Image
from torchvision import transforms as transforms
from torchvision import models as models
import torch.utils.data as data
import os
import torch.nn as nn
import torch.nn.functional as F
import torch

class dataset(data.Dataset):
    def __init__(self, root_path='F:\\数字图像处理\\DIP 2', mode='train', size=256, task=1):
        super(dataset, self).__init__()
        self.root_path = root_path
        self.task = task
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # resize = transforms.Resize((224, 224))
        resize = transforms.Resize((size, size))
        normalize = transforms.Normalize((0.1626,), (0.3356,))
        if mode == 'train':
            self.transform = transforms.Compose([resize,
                                                transforms.RandomRotation(15),
                                                transforms.ToTensor(),
                                                normalize])
        else:
            self.transform = transforms.Compose([resize,
                                                transforms.ToTensor(),
                                                normalize])
        if mode == 'train':
            self.txt_path = 'images_labels_train.txt'
        else:
            self.txt_path = 'images_labels_test.txt'
        
        f = open(self.txt_path)
        lines = f.readlines()
        f.close()
        self.x = [line.strip().split()[0] for line in lines]
        self.y = [int(line.strip().split()[1]) for line in lines]
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        # print(self.x[idx])
        x = Image.open(os.path.join(self.root_path, 'dataset', self.x[idx].split('\\')[0], self.x[idx].split('\\')[1]))#.convert('RGB')
        x = self.transform(x)
        x = 1 - x
        y = self.y[idx]
        if self.task == 2:
            return x, int(y/4)
        else:
            return x, y

class Alex(nn.Module):
    def __init__(self, num_classes=40):
        super(Alex, self).__init__()
        self.Conv1 = nn.Conv2d(1, 3, 3, 1)
        self.alex = models.resnet152(pretrained=False)
        self.fc = nn.Linear(1000, num_classes)
    
    def forward(self, x):
        x = self.Conv1(x)
        x = self.alex(x)
        # x = F.relu(x)
        x = self.fc(x)
        # x = F.softmax(x, dim=1)

        return x

class ConvNet(nn.Module):
    def __init__(self, num_classes=40):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(15488, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # x = F.log_softmax(x, dim=1)
        return x