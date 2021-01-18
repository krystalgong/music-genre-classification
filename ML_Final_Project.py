import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

def file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.h5':  
                L.append(os.path.join(root, file)[29:-3])  
    return L # file name = id

def open_genre_1(lst):
    f = open('genre.txt','r')
    dic = {}
    Set = []
    id_genre = {}
    for i in f.readlines():
        content = i.split()
        content[-1] = content[-1][:-1]
        if len(content) == 3:
            content[1] = content[1]+" "+content[2]
        if len(content) == 1:
            continue
        dic[content[0]] = content[1]
        Set.append(content[1])
    genre = list(set(Set))
    # print(genre)
    # print(len(set(Set)))
    number = [0]*15
    for j in lst:
        if j in dic:
            id_genre[j] = dic[j]
    for m in id_genre:
        number[genre.index(id_genre[m])] +=1
    # print(number)
    # print(sum(number))
    return id_genre, genre

def open_genre_2(lst,id_genre):
    g = open('newgenre.txt','r')
    dic2 = {}
    # Set2 = []
    for k in g.readlines():
        content = k.split()
        if len(content) == 3:
            content[1] = content[1]+" "+content[2]
        if len(content) == 1:
            continue
        dic2[content[0]] = content[1]
    #     Set2.append(content[1])
    # genre2 = list(set(Set2))
    # print(genre2)
    # print(len(set(Set2)))
    for item in lst:
        if (item not in id_genre) and (item in dic2) and dic2[item] != "Pop_Rock":
            id_genre[item] = dic2[item]
    # print(len(id_genre))
    return id_genre

def data_pre_process(updated_id_genre):
    processed_data = {}
    x = []
    y = []
    for k,v in updated_id_genre.items():
        lst = []
        f = h5py.File(k+'.h5', 'r')
        for i in f['analysis']:
            for j in f['analysis'][i]:
                try:
                    lst.append(float(j))
                except:
                    for k in list(j):
                        lst.append(k)
        Array = np.array(lst) # len(lst) = 29746
        new = np.reshape(Array[162:],(172,172))
        processed_data[k] = [v,new]
        x.append(new)
        y.append(v)
    return processed_data, x, y


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 16, 3)
        self.fc1 = nn.Linear(16 * 41 * 41, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 15)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def main():
    lst = file_name("MillionSongSubset/data")

    id_genre, genre = open_genre_1(lst)
    # genre = ['Rock', 'Country', 'New Age', 'Jazz', 'Folk', 'Blues', 'Pop', 'Metal', 'World', 'Reggae', 'Electronic', 'Punk', 'Rap', 'RnB', 'Latin']
    updated_id_genre = open_genre_2(lst,id_genre)

    processed_data, x, y = data_pre_process(updated_id_genre)

    x_train = x[:2344]
    x_validation = x[2344:2844]
    x_test = x[2844:]
    y_train = y[:2344]
    y_validation = y[2344:2844]
    y_test = y[2844:]








main()