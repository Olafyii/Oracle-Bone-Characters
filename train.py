import torch.utils.data as data
from utils import *
import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np
import os

def train(model, trainloader, optimizer, epoch, device):
    N_count = 0
    model.train()
    for batch_idx, (x, y) in enumerate(trainloader):
        N_count += x.size(0)
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        raw_y_pred = model(x)
        # loss = F.cross_entropy(y_pred, y)
        loss = F.cross_entropy(raw_y_pred, y)
        loss.backward()
        optimizer.step()

        y_pred = torch.max(raw_y_pred, 1)[1]
        # if (batch_idx+1)%20 == 0:
        #     print('yyyyyyyyyyyy', y)
        #     print('y_predy_pred', raw_y_pred)
        accu = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(trainloader.dataset), 100. * (batch_idx + 1) / len(trainloader), loss.item(), 100 * accu))

def validation(model, testloader, epoch, device):
    model.eval()
    all_y = []
    all_y_pred = []
    for batch_idx, (x, y) in enumerate(testloader):
        x, y = x.to(device), y.to(device)

        y_pred = model(x).max(1, keepdim=True)[1]

        all_y.extend(y)
        all_y_pred.extend(y_pred)
        print('Validating, epoch %d, batch_idx %d' % (epoch + 1, batch_idx))

    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    accu = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
    print('Accu: %f' % accu)
    return accu

def save_model(model, epoch):
    torch.save(model.state_dict(), os.path.join('models', 'epoch_%d.pth'%epoch))

if __name__ == '__main__':
    device = torch.device('cuda')

    model = Alex().to(device)
    trainset = dataset(mode='train', size=256)
    testset = dataset(mode='test', size=256)

    params = {'batch_size': 64, 'shuffle': True, 'num_workers': 0, 'pin_memory': False}

    trainloader = data.DataLoader(trainset, **params)
    testloader = data.DataLoader(testset, **params)

    optimizer = torch.optim.Adadelta(model.parameters(), lr=1)
    best_accu = 0
    for epoch in range(1000):
        train(model, trainloader, optimizer, epoch, device)
        accu = validation(model, testloader, epoch, device)
        if accu > best_accu:
            best_accu = accu
            if accu > 0.83:
                save_model(model, epoch)
    print('best_accu', best_accu)

