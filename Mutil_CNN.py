import os
import torch
import cv2
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class Net(nn.Module):

    def __init__(self, input_size, classes):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size[0], 16, kernel_size=5),# 图片大小：44*140
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)                             # 图片大小：22*70
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),           # 图片大小：20*68
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)                             # 图片大小：10*34
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),          # 图片大小：8*32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)                             # 图片大小：4*16
        )
        self.fc1 = nn.Sequential(
            nn.Linear(4*16*128, 4096),
            nn.Dropout(0.2),  
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.Dropout(0.2),  
            nn.ReLU()
        )
        self.fc3 = nn.Linear(2048, classes)

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
    label_list =[]
    first_vec, others_vec = vec.split([len(labels), (len(labels)-1) * (plate_len -1)],dim=1)

    first_vec = torch.argmax(first_vec, dim=1)
    label_list.append(labels[first_vec])

    others_vec = others_vec.view(plate_len-1,-1)
    others_vec = torch.argmax(others_vec, dim=1)
    for item in others_vec:
        label_list.append(labels[item])
    label = "".join(label_list)
    return label

net = Net(input_size=[1, 48, 144], classes = 61)
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A']
plate_len = 6


# 图像处理
def pre_img(img):
    resized = cv2.resize(img, (144, 48))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

    # blur = cv2.GaussianBlur(gray,(1,1),0)  #(3,3)为高斯半径
    # _, binary = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary

def get_label(model_path, crop_img):
    
    img_bin = pre_img(crop_img)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint)

    def predict(inputs):
        net.eval()
        with torch.no_grad():
            outputs = net(inputs)
            # outputs = outputs.view(-1, len(plate_list)) 
        return vec2text(outputs)

    transform = transforms.Compose([transforms.ToTensor()]) 
    vil_data = CaptchaData(img_bin, transform=transform)
    vil_data_loader = DataLoader(vil_data, batch_size=1, num_workers=4, shuffle=False)

    for img in vil_data_loader:
        pre = predict(img)

    return pre