import os, sys
import random
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import PIL, torchvision
print(pd.__version__=="0.25.2", pd.__version__)
print(np.__version__=="1.17.3", np.__version__)
print(PIL.__version__=="6.1.0", PIL.__version__)
print(torch.__version__=="1.2.0+cu92", torch.__version__)
print(torchvision.__version__=="0.4.0+cu92", torchvision.__version__)
print(torch.cuda.is_available())

class hw3_dataset(Dataset):
    
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx][0]).convert('RGB')
        img = self.transform(img)
        label = self.data[idx][1]
        return img, label

def load_data(img_path, label_path):
    train_image = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
    print(os.path.join(img_path, '*.jpg'))
    train_label = pd.read_csv(label_path)
    train_label = train_label.iloc[:,1].values.tolist()
    
    train_data = list(zip(train_image, train_label))
    random.shuffle(train_data)
    
    train_set = train_data[:28000]
    valid_set = train_data[28000:]


    print(len(train_set))
    print(len(valid_set))

    return train_set, valid_set

import torchvision.models as models

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.resnet = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        self.fc = nn.Sequential(nn.Dropout(p=0.5),nn.Linear(512,7))
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 1*1*512)
        x = self.fc(x)
        
        return x
if __name__ == '__main__':
    train_data_folder = sys.argv[1]
    train_csv = sys.argv[2]
    use_gpu = torch.cuda.is_available()
    train_set, valid_set = load_data(train_data_folder, train_csv)
    #transform to tensor, data augmentation
    
    transform = transforms.Compose([
    #transforms.RandomAffine(15, translate=(0.05,0.05), scale=(0.95,1.05), shear=1, fillcolor=0),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize([mean], [std], inplace=False)
    ])
    
    train_dataset = hw3_dataset(train_set,transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    valid_dataset = hw3_dataset(valid_set,transform)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

    #model = Net()
    model = Resnet18()
    #model.load_state_dict(torch.load('model_best.pth'))


    if use_gpu:
        model.cuda()

    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.4)
    loss_fn = nn.CrossEntropyLoss()

    num_epoch = 50
    train_loss_trace = []
    train_acc_trace = []
    valid_loss_trace = []
    valid_acc_trace = []
    for epoch in range(num_epoch):
        model.train()
        for param_group in optimizer.param_groups:
            print("lr:", param_group['lr'])
        pred = []
        labe = []
        train_loss = []
        train_acc = []
        for idx, (img, label) in enumerate(train_loader):
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            
            predict = torch.max(output, 1)[1]
            acc = np.mean((label == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())
        print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))
        scheduler.step()

        model.eval()
        with torch.no_grad():
            valid_loss = []
            valid_acc = []
            for idx, (img, label) in enumerate(valid_loader):
                if use_gpu:
                    img = img.cuda()
                    label = label.cuda()
                output = model(img)
                loss = loss_fn(output, label)
                predict = torch.max(output, 1)[1]
                acc = np.mean((label == predict).cpu().numpy())
                predict = predict.tolist()
                label = label.tolist()
                for j in predict:
                    pred.append(j)
                for j in label:
                    labe.append(j)
                valid_loss.append(loss.item())
                valid_acc.append(acc)
            print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}".format(epoch + 1, np.mean(valid_loss), np.mean(valid_acc)))
            
        #train_loss_trace.append(np.mean(train_loss))
        #train_acc_trace.append(np.mean(train_acc))
        #valid_loss_trace.append(np.mean(valid_loss))
        #valid_acc_trace.append(np.mean(valid_acc))
        #np.save("train_loss_trace", train_loss_trace)
        #np.save("train_acc_trace", train_acc_trace)
        #np.save("valid_loss_trace", valid_loss_trace)
        #np.save("valid_acc_trace", valid_acc_trace)
        #np.save("pred", pred)
        #np.save("labe", labe)


        if np.mean(train_acc) > 0.95:
            checkpoint_path = 'model_{}.pth'.format(epoch+1) 
            torch.save(model.state_dict(), checkpoint_path)
            print('model saved to %s' % checkpoint_path)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(1, 128, kernel_size=3,padding=1,bias=False),nn.ReLU(),nn.BatchNorm2d(128,affine=False),nn.AdaptiveMaxPool2d(output_size=(48, 48)))
        self.conv2=nn.Sequential(nn.Conv2d(128, 64, kernel_size=3,padding=1,bias=False),nn.ReLU(),nn.BatchNorm2d(64,affine=False),nn.AdaptiveMaxPool2d(output_size=(48, 48)))
        self.conv3=nn.Sequential(nn.Conv2d(64, 64, kernel_size=3,padding=1,bias=False),nn.ReLU(),nn.BatchNorm2d(64,affine=False),nn.AdaptiveMaxPool2d(output_size=(48, 48)))
        self.conv4=nn.Sequential(nn.Conv2d(64, 32, kernel_size=3,padding=1,bias=False),nn.ReLU(),nn.BatchNorm2d(32,affine=False),nn.AdaptiveMaxPool2d(output_size=(25, 25)))
        
        self.fc = nn.Sequential(nn.Linear(25*25*32, 512,bias=False),nn.ReLU(),nn.Dropout(p=0.5),nn.Linear(512, 7,bias=False))
    def forward(self, x):
        #image size (48,48)
        x = self.conv1(x) 
        x = self.conv2(x) 
        x = self.conv3(x) 
        x = self.conv3(x) 
        x = self.conv3(x) 
        x = self.conv3(x) 
        x = self.conv4(x) 
        x = x.view(-1, 25*25*32)
        x = self.fc(x)
        return x