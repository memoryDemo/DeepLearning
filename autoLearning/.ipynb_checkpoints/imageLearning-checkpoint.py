import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./data", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="./data", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)
# 利⽤dataloader来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 搭建神经⽹络
class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()
        self.model = nn.Sequential(
            # 卷积
            nn.Conv2d(3, 32, 5, 1, 2),
            # 最⼤池化
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            # 线性层
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 验证⽹络正确性
if __name__ == '__main__':
    train = NNModel()
    NNInput = torch.ones((64, 3, 32, 32))
    output = train(NNInput)
    print(output.shape)

# 创建⽹络模型
nnmodel = NNModel()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# learning_rate = 0.01
# 1e-2 = 1 * (10)^-2 = 1/100 = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(nnmodel.parameters(), lr=learning_rate)

# 设置训练⽹络的⼀些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的论述
epoch = 10

# 添加tensorboard
writer = SummaryWriter("./logs_train")

# 开始训练
for i in range(epoch):
    print("-----第 {} 轮训练开始-----".format(i + 1))
    # 训练步骤开始
    # 使神经⽹络进⼊训练状态
    # nnmodel.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = nnmodel(imgs)
        # optputs 预测的输出 ， targets 真实值
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播 --> 得到每⼀个参数节点的梯度
        loss.backward()
        # 调⽤优化器 进⾏优化
        optimizer.step()
        # 训练次数加1
        total_train_step += 1

        # 以特定的⽅式输出
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # nnmodel.eval()
    # 在训练完⼀次后，使⽤测试集进⾏测试
    total_test_loss = 0
    # 整体正确的个数
    total_accuracy = 0

    # 取消所有梯度
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = nnmodel(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

        print("整体测试集上的loss:{}".format(total_test_loss))
        print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("total_accuracy", total_accuracy / test_data_size, total_test_step)

        total_test_step += 1

        # 保存每次训练后的结果
        torch.save(nnmodel, "nnmodel_{}.pth".format(i))
        # 官⽅推荐的保存模型⽅式
        # torch.save(nnmodel.state_dict(),"nnmodel_{}.pth".format(i))
        print("模型以保存")

writer.close()
