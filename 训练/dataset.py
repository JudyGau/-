import csv
import os
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from pathlib import Path

batch_size = 5


def read_file(path):  # 读取数据，文件夹中包含五个子文件夹

    data_list = os.listdir(path)  # 得到5个子文件夹名称
    data_len = 0  # 存放总图片数目
    for type in data_list:
        data_len += len(os.listdir(os.path.join(path, type)))

    data = np.uint8(np.zeros((data_len, 224, 224, 3)))
    data_label = np.zeros(data_len)
    i = 0
    for j, type in enumerate(data_list):  # 读出所有图片
        list = os.listdir(os.path.join(path, type))
        for img in list:
            if img.split(".")[-1] == "ipynb_checkpoints":
                continue
            img_path = f"{path}/{type}/{img}"
            print(img_path)
            data[i, :, :, :] = cv2.resize(
                # cv2.imread(img_path),
                cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR),
                (224, 224),
            )
            data_label[i] = j
            i += 1
    return data, data_label, data_list


class ImgDataset(Dataset):
    def __init__(self, x=None, y=None, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


def process_data(data_path):
    data, label, label_list = read_file(data_path)
    transform = transforms.Compose(
        [  # 数据处理
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化处理
        ]
    )
    dataset = ImgDataset(data, label, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, label, label_list
