基于pytorch实现对验证码的识别
==========================
**Author**: `Shiyipaisizuo <https://github.com/Shiyipaisizuo/pytorch_project>`_。

一.前提需求
==========


* Python (>=3.4)
* sklearn (>= 0.19.0)
* PIL (>=5.0.0)
* torchvision (>=0.2.0)
* torch (>=0.3.0)
* numpy (>=1.14.2)
* captcha （>=0.2.4)
*在终端运行：*

**conda:**
::

    conda install -r requirements.txt
or **pip:**
::

    pip3 install -r requirements.txt

二.生成验证码
==============

这里我们用专业的captcha第三方库生成

**导入生成验证码所需库包**

::

    from captcha.image import ImageCaptcha
    import numpy as np
    from PIL import Image
    import random
    import cv2

**准备需要生成的随机数和字符**

::


    number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    alphabet = [
        'a',
        'b',
        'c',
        'd',
        'e',
        'f',
        'g',
        'h',
        'i',
        'j',
        'k',
        'l',
        'm',
        'n',
        'o',
        'p',
        'q',
        'r',
        's',
        't',
        'u',
        'v',
        'w',
        'x',
        'y',
        'z']
    ALPHABET = [
        'A',
        'B',
        'C',
        'D',
        'E',
        'F',
        'G',
        'H',
        'I',
        'J',
        'K',
        'L',
        'M',
        'N',
        'O',
        'P',
        'Q',
        'R',
        'S',
        'T',
        'U',
        'V',
        'W',
        'X',
        'Y',
        'Z']

    data_path = './data/'

**开始生成随机验证码图片**

::

    def gen_capthcha_text_and_image(m):
        image = ImageCaptcha()
        captcha_text = random_captcha_text()  # 生成数字
        captcha_text = ' '.join(captcha_text)  # 生成标签

        captcha = image.generate(captcha_text)

        captcha_image = Image.open(captcha)
        captcha_image = np.array(captcha_image)

        with open(data_path + "label.txt", "a") as f:  # 写入标签
            f.write(captcha_text)
            f.writelines("\n")
        cv2.imwrite(data_path + '%.5d.jpg' % m, captcha_image)  # 保存


**我们开始运行吧！**

::

    if __name__ == '__main__':

        for m in range(0, 10000):
            gen_capthcha_text_and_image(m)

**同样，你也可以随机设置只生成数字而不写入标签(不推荐)**

::

    def random_captcha_text(char_set=number, captcha_size=4):  # 可以设置只用来生成数字
        captcha_text = []
        for i in range(captcha_size):
            c = random.choice(char_set)
            captcha_text.append(c)
        return captcha_text


三.测试数据
==============

**对于一张验证码来说作为一张单一的图片，每输入一张图片，得到四个数字作为输出，只有4个数字同时预测正确才表示预测正确。所以在每一张图上是四个多二分类器：因为验证码上面的数字为0-9，只不过此时一张图片对应于4个数字。**

**同样，测试时导入所需库包**

::

    import copy
    import os
    import time
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image
    from torch.autograd import Variable
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms

**设置路径及加载数据**

::

    file_path = './data/'
    BATCH_SIZE = 16
    EPOCH = 10

    
    # Load data
    class DataSet(torch.utils.data.Dataset):

        def __init__(self, label_file, transform=None):
            self.label = np.loadtxt(label_file)
            self.transform = transform

        def __getitem__(self, idx):
            img_name = os.path.join('./data/%.4d.jpg' % idx)
            image = Image.open(img_name)
            labels = self.label[idx, :]

            #            sample = image

            if self.transform:
                image = self.transform(image)

            return image, labels

        def __len__(self):
            return self.label.shape[0]


    data = DataSet(file_path + 'label.txt', transform=transforms.ToTensor())

    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True)

    dataset_size = len(data)

**定义CNN卷积神经网络**

::

    # Conv network
    class CNN(nn.Module):

        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    3,
                    32,
                    kernel_size=4,
                    stride=1,
                    padding=2),
                # in:(bs,3,60,160)
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool2d(kernel_size=2),  # out:(bs,32,30,80)

                nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool2d(kernel_size=2),  # out:(bs,64,15,40)

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.MaxPool2d(kernel_size=2)  # out:(bs,64,7,20)
            )

            self.fc1 = nn.Linear(64 * 7 * 20, 500)
            self.fc2 = nn.Linear(500, 40)

        def forward(self, x):
            x = self.conv1(x)
            x = x.view(x.size(0), -1)  # reshape to (batch_size, 64 * 7 * 30)
            y = self.fc1(x)
            y = self.fc2(y)

            return y

**(可选)可以自定义损失函数**

::

    class nCrossEntropyLoss(nn.Module):

        def __init__(self, n=4):
            super(nCrossEntropyLoss, self).__init__()
            self.n = n
            self.total_loss = 0
            self.loss = nn.CrossEntropyLoss()

        def forward(self, ot, labels):
            output_t = ot[:, 0:10]
            labels = Variable(torch.LongTensor(labels.data.cpu().numpy()))
            label_t = labels[:, 0]

            for j in range(1, self.n):
                output_t = torch.cat((output_t, ot[:, 10 * j:10 * j + 10]),
                                     0)  # 损失的思路是将一张图平均剪切为4张小图即4个多分类，然后再用多分类交叉熵方损失
                label_t = torch.cat((label_t, labels[:, j]), 0)
                self.total_loss = self.loss(output_t, label_t)

            return self.total_loss

**开始测试吧。**

::

    def equal(np1, np2):
        n = 0
        for k in range(np1.shape[0]):
            if (np1[k, :] == np2[k, :]).all():
                n += 1

        return n


    net = CNN()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0

    since = time.time()
    for epoch in range(EPOCH):

        running_loss = 0.0
        running_corrects = 0

        for step, (inputs, label) in enumerate(data_loader):

            pred = torch.LongTensor(BATCH_SIZE, 1).zero_()
            inputs = Variable(inputs)  # (bs, 3, 60, 240)
            label = Variable(label)  # (bs, 4)

            optimizer.zero_grad()

            output = net(inputs)  # (bs, 40)
            loss = loss_func(output, label)

            for i in range(4):
                pre = F.log_softmax(
                    output[:, 10 * i:10 * i + 10], dim=1)  # (bs, 10)
                predection = torch.cat(
                    (predection, pre.data.max(
                        1, keepdim=True)[1].cpu()), dim=1)  #

            loss.backward()
            optimizer.step()

            running_loss += loss.data[0] * inputs.size()[0]
            running_corrects += equal(pred.numpy()
                                      [:, 1:], label.data.cpu().numpy().astype(int))

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects / dataset_size

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(net.state_dict())

        if epoch == EPOCH - 1:
            torch.save(best_model_wts, file_path + '/model.pkl')

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Train Loss:{:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

**随机生成的10000章图片中，正确率达到了98.5%，还算可以。如果能随机生成带英文的验证码就更好了。**