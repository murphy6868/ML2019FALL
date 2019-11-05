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

def load_data(img_path):
    test_image = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
    return test_image
class hw3_dataset(Dataset):
    
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        img = self.transform(img)
        return img
if __name__ == '__main__':
    path_ans = sys.argv[2]
    test_data_folder = sys.argv[1]
    use_gpu = torch.cuda.is_available()
    test_image = load_data(test_data_folder)
    transform = transforms.Compose([
    #transforms.RandomAffine(15, translate=(0.1,0.1), scale=(0.9,1.1), shear=10, fillcolor=0),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize([mean], [std], inplace=False)
    ])
    test_dataset = hw3_dataset(test_image,transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


    model = Resnet18()
    model.load_state_dict(torch.load('model_best.pth'))
    if use_gpu:
        model.cuda()
    model.eval()
    with torch.no_grad():
        result = []
        for idx, (img) in enumerate(test_loader):
            if use_gpu:
                img = img.cuda()
            output = model(img)
            predict = torch.max(output, 1)[1]
            predict = predict.cpu().numpy().tolist()
            for j in predict:
                result.append(j)

        ans = pd.DataFrame({"id" : range(0, len(result))})
        ans['label'] = result
        print(ans)
        ans.to_csv(path_ans, index = False)