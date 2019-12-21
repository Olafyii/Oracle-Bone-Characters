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
    Y_pred = []
    Y_true = []
    Loss = []
    print('len(trainloader)', len(trainloader))
    for batch_idx, (x, y) in enumerate(trainloader):
        Y_true += list(y.numpy())  # stat
        N_count += x.size(0)
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        raw_y_pred = model(x)
        loss = F.cross_entropy(raw_y_pred, y)
        print(loss.item())
        Loss += [loss.item()]  # stat
        loss.backward()
        optimizer.step()
        y_pred = torch.max(raw_y_pred, 1)[1]
        Y_pred += list(y_pred.numpy())  # stat
        accu = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(trainloader.dataset), 100. * (batch_idx + 1) / len(trainloader), loss.item(), 100 * accu))
    accu = accuracy_score(Y_true, Y_pred)
    loss = sum(Loss)*1.0 / len(Loss)
    return accu, loss


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

def save_model(model, epoch, save_path):
    torch.save(model.state_dict(), os.path.join('models/resnet', 'epoch_%d.pth'%epoch))

if __name__ == '__main__':
    save_model_path = 'models/resnet152_Adadelta_Task2'
    save_accu_path = 'losses/resnet152_Adadelta_Task2'
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    if not os.path.exists(save_accu_path):
        os.mkdir(save_accu_path)

    root_path = '/lustre/home/acct-cszlq/cszlq/lwk/Oracle-Bone-Characters'
    # root_path = 'F:\\数字图像处理\\DIP 2'

    device = torch.device('cuda')

    model = Alex(10).to(device)
    trainset = dataset(root_path=root_path, mode='train', size=256, task=2)
    testset = dataset(root_path=root_path, mode='test', size=256, task=2)

    params = {'batch_size': 64, 'shuffle': True, 'num_workers': 0, 'pin_memory': False}

    trainloader = data.DataLoader(trainset, **params)
    testloader = data.DataLoader(testset, **params)

    optimizer = torch.optim.Adadelta(model.parameters(), lr=1)
    best_accu = 0

    train_accu_all = []
    train_loss_all = []
    test_accu_all = []
    for epoch in range(1000):
        train_accu, loss = train(model, trainloader, optimizer, epoch, device)
        test_accu = validation(model, testloader, epoch, device)

        train_accu_all += train_accu
        train_loss_all += loss
        test_accu_all += test_accu

        A = np.array(train_accu_all)
        B = np.array(train_loss_all)
        C = np.array(test_accu_all)

        np.save(os.path.join(save_accu_path, 'train_accu.npy'), A)
        np.save(os.path.join(save_accu_path, 'train_loss.npy'), B)
        np.save(os.path.join(save_accu_path, 'test_accu.npy'), C)

        if test_accu > best_accu:
            best_accu = test_accu
            save_model(model, epoch, save_model_path)
    print('best_accu', best_accu)

