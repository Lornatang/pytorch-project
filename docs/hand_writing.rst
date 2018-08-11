基于pytorch实现对手写字的识别
==========================
**Author**: `Shiyipaisizuo <https://github.com/Shiyipaisizuo/pytorch_project>`_。

一.前提需求
=================


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

二.准备前提
==============================

**导入相应的库包同时开启交互模式**

::

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.autograd import Variable
    from torch.utils.data import DataLoader as DataLoader

    torch.manual_seed(1)

三.数据处理
=================

**下载训练数据和测试数据**

::

	train_dataset = datasets.MNIST(root='./data/',
	                               train=True,
	                               transform=transforms.ToTensor(),
	                               download=False)

	test_dataset = datasets.MNIST(root='./data/',
	                              train=False,
	                              transform=transforms.ToTensor())

**加载训练数据和测试数据**

::

	train_loader = DataLoader(dataset=train_dataset,
	                          batch_size=64,
	                          shuffle=True)

	test_loader = DataLoader(dataset=test_dataset,
	                         batch_size=64,
	                         shuffle=False)

四.神经网络
=================

**CNN神经网络**

::

	class CNN(nn.Module):
	    def __init__(self):
	        super(CNN, self).__init__()
	        # 输入1通道，输出10通道，kernel 5*5
	        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
	        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
	        self.mp = nn.MaxPool2d(2)
	        # fully connect
	        self.fc = nn.Linear(320, 10)

	    def forward(self, x):
	        # in_size = 64
	        in_size = x.size(0)  # one batch
	        # x: 64*10*12*12
	        x = F.relu(self.mp(self.conv1(x)))
	        # x: 64*20*4*4
	        x = F.relu(self.mp(self.conv2(x)))
	        # x: 64*320
	        x = x.view(in_size, -1)  # flatten the tensor
	        # x: 64*10
	        x = self.fc(x)
	        return F.log_softmax(x)


**定义优化器**
::

	cnn = CNN()

	optimizer = optim.SGD(cnn.parameters(), lr=0.01, momentum=0.5)


五.训练和测试
=================

**训练函数**

::

	def train(x):
	    for batch_idx, (data, target) in enumerate(train_loader):
	        data, target = Variable(data), Variable(target)
	        optimizer.zero_grad()
	        output = cnn(data)
	        loss = F.nll_loss(output, target)
	        loss.backward()
	        optimizer.step()
	        if batch_idx % 200 == 0:
	            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
	                x, batch_idx * len(data), len(train_loader.dataset),
	                100. * batch_idx / len(train_loader), loss.data[0]))

**测试函数**

::

	def test():
	    test_loss = 0
	    correct = 0
	    for data, target in test_loader:
	        data, target = Variable(data, volatile=True), Variable(target)
	        output = cnn(data)
	        # sum up batch loss
	        test_loss += F.nll_loss(output, target, size_average=False).data[0]
	        # get the index of the max log-probability
	        pred = output.data.max(1, keepdim=True)[1]
	        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

	    test_loss /= len(test_loader.dataset)
	    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
	        test_loss, correct, len(test_loader.dataset),
	        100. * correct / len(test_loader.dataset)))

六.运行
=============

::

	for epoch in range(1, 10):
	    train(epoch)
	    test()