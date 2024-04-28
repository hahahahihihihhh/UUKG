import numpy as np
import torch
import torch.nn as nn


# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.fc = nn.Linear(in_features=192, out_features=768)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(in_features=768, out_features=192)
#
#     def forward(self, x):
#         x = self.fc(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x
#
#
# def huber_loss(preds, labels, delta=1.0):
#     residual = torch.abs(preds - labels)
#     condition = torch.le(residual, delta)
#     small_res = 0.5 * torch.square(residual)
#     large_res = delta * residual - 0.5 * delta * delta
#     return torch.mean(torch.where(condition, small_res, large_res))
#
#
# def func(x: torch.Tensor) -> torch.FloatTensor:
#     """
#     f(x) = 2x + 365
#     """
#     return x * 2 + 365
#
#
# if __name__ == '__main__':
#     train_x = [
#         torch.tensor([_ for _ in range(0, 384, 2)], dtype=torch.float32),
#         torch.tensor([_ for _ in range(382, -1, -2)], dtype=torch.float32),
#         torch.tensor([_ for _ in range(384, 768, 2)], dtype=torch.float32),
#         torch.tensor([_ for _ in range(767, 383, -2)], dtype=torch.float32),
#         torch.tensor([_ for _ in range(768, 384, -2)], dtype=torch.float32),
#         torch.tensor([_ for _ in range(383, 0, -2)], dtype=torch.float32),
#         torch.tensor([_ for _ in range(385, 2, -2)], dtype=torch.float32),
#     ]
#     train_y = [
#         func(x) for x in train_x
#     ]
#     test_x = torch.tensor(
#         [_ for _ in range(1, 384, 2)], dtype=torch.float32
#     )
#     test_y = func(test_x)
#     mynet = MyModel()
#     mynet.train()
#     optimizer = getattr(torch.optim, "Adam")(params=mynet.parameters(), lr=0.0001)  # Adam优化器
#     for epoch in range(2000):
#         losses = []
#         for x, y in zip(train_x, train_y):
#             # mini-batch
#             optimizer.zero_grad()  # 将梯度初始化为零
#             _y = mynet(x)  # 前向传播求出预测的值
#             loss = huber_loss(_y, y)  # 求loss
#             loss.backward()  # 反向传播求梯度
#             optimizer.step()  # 更新所有参数
#             losses.append(loss.detach().numpy())
#         print("Epoch {}, Loss: {}".format(epoch, np.mean(losses)))
#     with torch.no_grad():
#         mynet.eval()
#         y = mynet(test_x)
#         loss = huber_loss(y, test_y)
#         print("Test loss: {}".format(loss))

# x = torch.tensor([2.0], requires_grad = True)
# b = torch.tensor([1.0], requires_grad = True)
# # print(b)
# y = x * x * x + b
# y.backward()
# print(x.grad_fn, y.grad_fn, b.grad_fn)

# x = torch.tensor([[1, 2], [3, 4], [5, 6]])
# print(x * x)
# x2 = torch.sum(x * x, dim=-1)
# print(x, x2)

# a = np.random.randint(
#             2,
#             size = 4)
# print(type(a))
# print(bh(torch.LongTensor([1, 1])))

# print(bh.weight)
# print(bh(torch.tensor([1, 2])))

# these_queries = torch.tensor([[1, 2, 3],
#                         [2, 3, 4]], dtype=torch.int32)
# scores = torch.LongTensor(([1, 2, 3, 4]))
# scores2 = torch.LongTensor([1, 2, 3, 4])
# print(scores, scores2, scores == scores2)
# a形状(2,3)
# a = torch.tensor([[1, 2, 3],
#                   [1, 2, 3]])
# # repeat参数比维度多，在扩展前先讲a的形状扩展为(1,2,3)然后复制
# b = a.repeat(2, 2, 2)
# print(b)  # 得到结果torch.Size([1, 4, 3])
# 生成一个字典
# dict = {'name': '','age': '','sex': ''}
# # 打印返回值
# print(dict.has_key('name'))  # 结果返回True
# print(dict.has_key('id'))  # 结果返回False

# t = torch.tensor([[1, 2],
#               [3, 4]], dtype=torch.float32)
# inputs1 = torch.norm(t, p=2, dim=1, keepdim=True)
# print(inputs1)


# a = torch.ones((1, 1))
# print(a)
# print(a[torch.tensor([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])])

a = torch.tensor([[1., 2.],
                  [2., 3.]])
b = torch.tensor([2., 3.])
print(a / b)