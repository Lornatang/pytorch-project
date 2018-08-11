利用pytorch实现图片分类
=====================
**Author**: `Shiyipaisizuo <https://github.com/Shiyipaisizuo/pytorch_project>`_。

一.前提需求
==========


* Python (>=3.4)
* sklearn (>= 0.19.0)
* PIL (>=5.0.0)
* torchvision (>=0.2.0)
* torch (>=0.3.0)

*在终端运行：*

**conda:**
::

    conda install -r requirements.txt
or **pip:**
::

    pip3 install -r requirements.txt

二.获取数据
===========

**数据我已经上传到了我的Github中，保存路径在`./data/`下**

三.转化二进制图片作为训练和测试集
==============================

::

    import os
    from skimage import io
    import torchvision.datasets.mnist as mnist

    root = "./classification/data/"  # loader_data path
    train_set = (
        mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte')))  # train_data path
    test_set = (
        mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte')))  # test_data path
    print("training set :", train_set[0].size())
    print("test set :", test_set[0].size())


    # convert(binary_to_image)
    def convert_to_img(train=True):
        if train:
            f = open('./classification/data/train.txt', 'w')
            data_path = './train/'
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
                img_path = data_path + str(i) + '.jpg'
                io.imsave(img_path, img.numpy())
                f.write(img_path + ' ' + str(label) + '\n')
            f.close()
        else:
            f = open('./classification/data/test.txt', 'w')
            data_path = './test/'
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
                img_path = data_path + str(i) + '.jpg'
                io.imsave(img_path, img.numpy())
                f.write(img_path + ' ' + str(label) + '\n')
            f.close()


    convert_to_img(True)  # train
    convert_to_img(False)  # test


四.训练及测试
==============

**先要将图片读取出来，准备成torch专用的dataset格式，再通过Dataloader进行分批次训练。**

::

    import torch
    from torch.autograd import Variable
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
    import torch.nn as nn

    root = './data'


    def default_loader(path):
        return Image.open(path).convert('RGB')


    class MyDataSet(DataLoader):

        def __init__(
                self,
                txt,
                transform=None,
                target_transform=None,
                loader=default_loader):
            fh = open(txt, 'r')
            imgs = []
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                imgs.append((words[0], int(words[1])))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            fn, label = self.imgs[index]
            img = self.loader(fn)
            if self.transform is not None:
                img = self.transform(img)
            return img, label

        def __len__(self):
            return len(self.imgs)


    train_data = MyDataSet(
        txt='./classification/data/train.txt',
        transform=transforms.ToTensor())
    test_data = MyDataSet(
        txt='./classification/data/test.txt',
        transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=64)

**在设计复杂的神经网络之前，我们依然考虑按照斯坦福大学的“UFLDL Tutorial”的CNN部分来构建一个简单的卷积神经网络，即按照以下的设计：**

::

    class Cnn(nn.Module):

        def __init__(self):
            super(Cnn, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.dense = nn.Sequential(
                nn.Linear(64 * 3 * 3, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )

        def forward(self, x):
            conv1_out = self.conv1(x)
            conv2_out = self.conv2(conv1_out)
            conv3_out = self.conv3(conv2_out)
            res = conv3_out.view(conv3_out.size(0), -1)
            out = self.dense(res)

            return out


    model = Cnn()
    print(model)

    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            train_loss += loss.data[0]
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
            train_data)), train_acc / (len(train_data))))

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = Variable(
                batch_x, volatile=True), Variable(
                batch_y, volatile=True)
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.data[0]
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.data[0]
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            test_data)), eval_acc / (len(test_data))))


会在屏幕打印出如下显示

::

    Cnn(
  (conv1): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
  )
  (conv2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
  )
  (conv3): Sequential(
    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
  )
  (dense): Sequential(
    (0): Linear(in_features=576, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=10, bias=True)
  )
    )

我用的是CPU跑数据，跑了比较久，大家可以使用GPU跑。

::

    epoch 1
    Train Loss: 0.008313, Acc: 0.805767
    Test Loss: 0.005751, Acc: 0.867700
    epoch 2
    Train Loss: 0.005027, Acc: 0.883567
    Test Loss: 0.004704, Acc: 0.890800
    epoch 3
    Train Loss: 0.004237, Acc: 0.900650
    Test Loss: 0.004431, Acc: 0.897100
    epoch 4
    Train Loss: 0.003821, Acc: 0.910583
    Test Loss: 0.004166, Acc: 0.898900
    epoch 5
    Train Loss: 0.003438, Acc: 0.919200
    Test Loss: 0.004462, Acc: 0.895600
    epoch 6
    Train Loss: 0.003168, Acc: 0.924700
    Test Loss: 0.003929, Acc: 0.909200
    epoch 7
    Train Loss: 0.002903, Acc: 0.931200
    Test Loss: 0.003961, Acc: 0.909800
    epoch 8
    Train Loss: 0.002646, Acc: 0.937533
    Test Loss: 0.003816, Acc: 0.912400
    epoch 9
    Train Loss: 0.002422, Acc: 0.942317
    Test Loss: 0.003898, Acc: 0.916100
    epoch 10
    Train Loss: 0.002204, Acc: 0.946950
    Test Loss: 0.003970, Acc: 0.912600
