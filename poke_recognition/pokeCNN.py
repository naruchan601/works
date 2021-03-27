## ポケモンのタイプ推定システムの作成

import cv2
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
import pprint
import torch
import torchvision
import torch.nn as nn
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as f
from torch.utils.data import DataLoader
from tqdm import tqdm


wb = xlrd.open_workbook('ポケモン高さ重さ.xlsx')
sheet = wb.sheet_by_name('Sheet1')

type_list = ['ノーマル', 'ほのお', 'みず', 'でんき', 'くさ', 'こおり', 'かくとう', 'どく', 'じめん', 'ひこう', 'エスパー', 'むし', 'いわ', 'ゴースト', 'ドラゴン', 'あく', 'はがね', 'フェアリー']
mini_type_list = ['ノーマル', 'ほのお', 'みず', 'でんき', 'くさ', 'どく', 'じめん', 'むし']



class Creat_Datasets(Dataset):
    def __init__(self, dir_name, data_transform):
        self.dir_name = dir_name
        self.file = os.listdir(dir_name)
        self.data_transform = data_transform
        
    def __len__(self):
        return len(self.file)
    
    def __getitem__(self, i):
        image = Image.open(self.dir_name + '/' + self.file[i]).convert('RGB')
        image = self.data_transform(image)
        label = torch.tensor(type_list.index(sheet.cell(int(self.file[i][:3]), 4).value))
        return image, label

class MyCNN(torch.nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)

        # self.pool = torch.nn.MaxPool2d(2, 2)  # カーネルサイズ, ストライド
        
        self.dropout1 = torch.nn.Dropout2d(p=0.1)

        self.fc1 = torch.nn.Linear(64 * 4 * 4, 18)  # 入力サイズ, 出力サイズ
        # self.dropout2 = torch.nn.Dropout(p=0.5)
        # self.fc2 = torch.nn.Linear(120, 84)
        # self.fc3 = torch.nn.Linear(84, 18)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = f.relu(self.conv4(x))
        x = f.relu(self.conv5(x))
        
        x = self.dropout1(x)
        x = x.view(-1, 64 * 4 * 4)  # 1次元データに変えて全結合層へ
        x = self.fc1(x)
        # x = self.dropout2(x)
        # x = f.relu(self.fc2(x))
        # x = self.fc3(x)
        # x = x / torch.sum(x)

        return x

if __name__ == '__main__':
    epoch = 300

    # loader = load_cifar10()
    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
            #    'dog', 'frog', 'horse', 'ship', 'truck')
    # data_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # ])

    data_transform = transforms.Compose([
        # transforms.RandomResizedCrop(64),
        transforms.Resize((128,128)),             # 256*256にリサイズ
        transforms.RandomHorizontalFlip(),  # ランダムで左右反転
        transforms.ToTensor(),               # テンソル化 [0~1]にスケーリング
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # length, 
    # img = np.array( Image.open('10_pokemon_image/001.jpg'))
    # print(img)
    # plt.imshow(img)
    # plt.show()
    # exit()
    datasets = Creat_Datasets('out', data_transform)
    data_loader = DataLoader(datasets, batch_size=1, shuffle=True)
    # print(data_loader)

    testsets = Creat_Datasets('trim_pokemon_image_151', data_transform)
    test_loader = DataLoader(testsets, batch_size=1, shuffle=True)

    test2sets = Creat_Datasets('trim_pokemon_image_251', data_transform)
    test2_loader = DataLoader(test2sets, batch_size=1, shuffle=True)
    # print(enumerate(data_loader))
    # tensor_image = enumerate(data_loader[0])
    # real_image = tensor_image.view(tensor_image.shape[1], tensor_image.shape[2], tensor_image.shape[0])
    # plt.imshow(real_image)
    # plt.show()
    # exit()
    
    # print(length)
    # print(datasets[1][2][300][500:600])
    # print(int(torch.ones([1])))
    # image = Image.open("pokemon_image/001.jpg")
    # print(image)
    # # exit()

    net: MyCNN = MyCNN()
    criterion = torch.nn.CrossEntropyLoss()  # ロスの計算
    optimizer = torch.optim.SGD(params=net.parameters(), lr=0.0001, momentum=0.9)

    # 学習前のフィルタの可視化
    # net.plot_conv1()
    # net.plot_conv2()

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'test2_acc': []
    }
    # type_list = ['ノーマル', 'ほのお', 'みず', 'でんき', 'くさ', 'こおり', 'かくとう', 'どく', 'じめん', 'ひこう', 'エスパー', 'むし', 'いわ', 'ゴースト', 'ドラゴン', 'あく', 'はがね', 'フェアリー']
    # mini_type_list = ['ノーマル', 'ほのお', 'みず', 'でんき', 'くさ', 'どく', 'じめん', 'むし']
    # mini_type_list = ['ほのお', 'みず', 'くさ', 'むし']

    # wb = xlrd.open_workbook('ポケモン高さ重さ.xlsx')
    # sheet = wb.sheet_by_name('Sheet1')
    
    # indices = sheet.nrows - 1
    # img = cv2.imread('pokemon_image/' + '{:0=3}'.format(poke_index + 1) + '.jpg')

    best_accuracy = 0.0
    n_waits = 0
    max_epoch = 0
    n_epochs = 0
    pred_type = np.zeros(151)
    best_pred = np.zeros(151)

    for e in range(epoch):
        net.train()
        loss = None
        sum_loss = 0
        correct = 0
        for i, (images, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            output = net(images)
            
            # labels = torch.tensor([type_list.index(sheet.cell(i+1, 4).value)])

            loss = criterion(output, labels)
            # print(output)
            # print(labels)
            # exit()
            # if i == 1:
            #     print(output)
            #     print(loss)
            #     print(images)
            #     tensor_image = images[0]
            #     real_image = tensor_image.view(tensor_image.shape[1], tensor_image.shape[2], tensor_image.shape[0])
                
            #     # image_numpy = real_image.to('cpu').detach().numpy().copy()
            #     # image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2HSV)
            #     # print(image_numpy)
            #     plt.imshow(real_image)
            #     plt.show()
            #     # images = images.to('cpu').detach().numpy().copy()
            #     # cv2.imshow('sample', images[0])
            #     # cv2.waitKey(0)
            #     # cv2.destroyAllWindows()
            #     exit()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).sum().item()
            # print(predicted[0].numpy()) 
            # pred_type[i] = predicted[0].numpy()
            # print(pred_type[i])
            # exit()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            

            if i % 100 == 0:
                print('Training log: {} epoch ({} / 4932 train. data). Loss: {}'.format(e + 1,
                                                                                         i + 1,
                                                                                         loss.item())
                      )
        print('{} acc:{}/{} ({}%) mean loss:{}'.format(e, correct, i+1, correct*100/(i+1), sum_loss/(i+1)))
        # # 学習過程でのフィルタの可視化
        # net.plot_conv1(e+1)
        # net.plot_conv2(e+1)

        history['train_loss'].append(loss.item())

        net.eval()
        correct = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(data_loader)):
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                # out_type[i] = outputs
                # if predicted == labels :
        
        print('{} acc:{}/{} ({}%) '.format(e, correct, i+1, correct*100/(i+1)))

        acc = float(correct / 4932)
        history['train_acc'].append(acc)

        correct = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(test2_loader)):
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        print('{} acc:{}/{} ({}%) '.format(e, correct, i+1, correct*100/(i+1)))
        acc = float(correct / 100)
        history['test2_acc'].append(acc)

        correct = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(test_loader)):
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                pred_type[i] = predicted[0].numpy()
                correct += (predicted == labels).sum().item()
        print('{} acc:{}/{} ({}%) '.format(e, correct, i+1, correct*100/(i+1)))
        acc = float(correct / 151)
        history['test_acc'].append(acc)
        n_epochs = e + 1
        if acc > best_accuracy:
            best_accuracy = acc
            best_pred = pred_type
            torch.save(net, 'model')
            n_waits = 0
            max_epoch = e
            # n_epochs = e + 1
        else:
            n_waits += 1
            # if n_waits > 50:
            #     # n_epochs = e + 1
            #     print("now:" + str(n_epochs) + " max_epoch:" + str(max_epoch) + " max_accuracy:" + str(best_accuracy))
            #     break
    print(" max_epoch:" + str(max_epoch) + " max_accuracy:" + str(best_accuracy))

    # 結果をプロット
    plt.plot(range(1, n_epochs + 1), history['train_loss'])
    plt.title('Training Loss [pokemon_type]')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('poke_img/type_loss.png')
    plt.close()

    plt.plot(range(1, n_epochs + 1), history['train_acc'], label='train_acc')
    plt.plot(range(1, n_epochs + 1), history['test_acc'], label='test_acc')
    plt.title('Accuracies [pokemon_type]')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('poke_img/type_acc.png')
    plt.close()

    # plt.plot(range(1, n_epochs + 1), history['train_acc'], label='train_acc')
    plt.plot(range(1, n_epochs + 1), history['test2_acc'], label='test2_acc')
    plt.title('Accuracies [pokemon_type]')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('poke_img/type_acc2.png')
    plt.close()

    text_name = 'result.txt'
    print(best_pred)
    with open(text_name, mode='w') as f:
        f.write('------text------\n')

    
    with open(text_name, mode='a') as f:
        f.write(str(best_pred))


    testsets = Creat_Datasets('trim_pokemon_image_151', data_transform)
    test_loader = DataLoader(testsets, batch_size=1, shuffle=False)

    net = torch.load('model')
    correct = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            pred_type[i] = predicted[0].numpy()
            correct += (predicted == labels).sum().item()
    print('0 acc:{}/{} ({}%) '.format(correct, i+1, correct*100/(i+1)))
    acc = float(correct / 151)

    text2_name = 'result2.txt'
    print(pred_type)
    with open(text2_name, mode='w') as f:
        f.write('------text------\n')

    
    with open(text2_name, mode='a') as f:
        f.write(str(pred_type))

