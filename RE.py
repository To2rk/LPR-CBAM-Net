import os
import torch
import cv2
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 第一层神经网络
        # nn.Sequential: 将里面的模块依次加入到神经网络中
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # 3通道变成16通道，图片大小：44*140
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)                             # 图片大小：22*70
        )
        # 第2层神经网络
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),           # 16通道变成64通道，图片大小：20*68
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)                             # 图片大小：10*34
        )
        # 第3层神经网络
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),          # 16通道变成64通道，图片大小：8*32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)                             # 图片大小：4*16，通道数 128
        )
        # 第4层神经网络
        self.fc1 = nn.Sequential(
            # in_features由输入张量的形状决定，out_features则决定了输出张量的形状 
            # nn.Linear(in_features, out_features)
            # out_feature：全连接层的神经元个数
            nn.Linear(4*16*128, 4096),
            nn.Dropout(0.4),  
            nn.ReLU()
        )
        # 第5层神经网络
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.Dropout(0.2),  
            nn.ReLU()
        )
        # 第6层神经网络
        self.fc3 = nn.Linear(1024, 6*11)                 # 6:验证码的长度， 11: 字母列表的长度

    #前向传播
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class CaptchaData(Dataset):
    def __init__(self, img_open, transform=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.img_open = img_open

    def __len__(self):
        return 1

    def __getitem__(self, _):

        if self.transform is not None:
            img = self.transform(self.img_open)
        return img

# 车牌向量转为文本
def vec2text(vec):
    label = torch.nn.functional.softmax(vec, dim =1)
    vec = torch.argmax(label, dim=1)
    for v in vec:
        text_list = [plate_list[v] for v in vec]
    return ''.join(text_list)

net = Net()
plate_list = list('0123456789A')
plate_length = 6

def get_label(model_path, crop_img):
    
    crop_img = cv2.resize(crop_img, (140, 44))

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model_state_dict'])

    def predict(inputs):
        net.eval()
        with torch.no_grad():
            outputs = net(inputs)
            outputs = outputs.view(-1, len(plate_list)) 
        return vec2text(outputs)

    transform = transforms.Compose([transforms.ToTensor()]) 
    vil_data = CaptchaData(crop_img, transform=transform)
    vil_data_loader = DataLoader(vil_data, batch_size=1, num_workers=0, shuffle=False)

    for img in vil_data_loader:
        pre = predict(img)

    return pre