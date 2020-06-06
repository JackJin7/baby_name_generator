# f = open('baby names/male.txt', 'r')
# for line in f.readlines():
#     if line.strip('\n') == '' or '#' in line:
#         continue
#     print(line, end='')
#     line = line.strip('\n').split('\t')
#     print(line)
# f.close()



# files = ['female.txt', 'male.txt', 'pet.txt']
# longest = 0
# data = []
# len_count = {}
# for file in files:
#     f = open('baby names/{}'.format(file), 'r')
#     for line in f.readlines():
#         line = line.strip('\n')
#         if line == '' or '#' in line:
#             continue
#         line = line.split('\t')
#         if ' ' in line[0]:
#             continue
#         if len(line[0]) not in len_count:
#             len_count[len(line[0])] = 0
#         len_count[len(line[0])] += 1
#     f.close()
# for i in range(2, 16):
#     print(i, len_count[i])

# s = 'Helen-Elizabeth'
# print(s.lower())


# data = [
#     torch.tensor([1, 2, 3, 4, 5]),
#     torch.tensor([1, 2, 3, 4]),
#     torch.tensor([1, 2, 3])
# ]
# print(data)
# data = rnn.pad_sequence(data, padding_value=0)
# print(data)
# data = rnn.pack_padded_sequence(data, [5, 4, 3])
# print(data)
# data = rnn.pad_packed_sequence(data, batch_first=True)
# print(data)

# prev = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
# a = torch.tensor([3, 2, 4])
# a = a.repeat(5).reshape(5, 3).t()
# print(a)
#
# a = torch.where(a <= 2, torch.zeros_like(prev), prev)
# print(a)

# import torch
# import torch.utils.data as Data
#
# torch.manual_seed(1)  # reproducible
#
# BATCH_SIZE = 8  # 每个batch的大小，取5或者8
#
# # 生成测试数据
# x = torch.linspace(0, 9, 10)  # x(torch tensor)
# y = torch.linspace(9, 0, 10)  # y(torch tensor)
#
# # 将输入和输出封装进Data.TensorDataset()类对象
# torch_dataset = Data.TensorDataset(x, y)
#
# # 把 dataset 放入 DataLoader
# loader = Data.DataLoader(
#     dataset=torch_dataset,  # 数据，封装进Data.TensorDataset()类的数据
#     batch_size=BATCH_SIZE,  # 每块的大小
#     shuffle=True,  # 要不要打乱数据 (打乱比较好)
#     num_workers=2,  # 多进程（multiprocess）来读数据
# )
#
# if __name__ == '__main__':  # 注意：如果loader中设置了num_workers!=0，即采用多进程来处理数据，运行含loader的操作必须在‘__main__’的范围内
#
#     # 进行3轮训练（每次拿全部的数据进行训练）
#     for epoch in range(3):
#         # 在一轮中迭代获取每个batch（把全部的数据分成小块一块块的训练）
#         for step, (batch_x, batch_y) in enumerate(loader):
#             # 假设这里就是你训练的地方...
#
#             # print出来一些数据
#             print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
#                   batch_x, '| batch y: ', batch_y)


# import torch
# import torch.nn as nn
#
# predicts = torch.tensor([
#     [1, 2, 3],
#     [2, 3, 4],
# ]).float()
# target = torch.tensor([
#     [0, 0, 1],
#     [0, 1, 0],
# ]).float()
# target = torch.argmax(target, dim=1)
# entroy = nn.CrossEntropyLoss(reduction='sum')
#
# loss = entroy(predicts, target)
# print(loss)
#
# predicts = torch.tensor([
#     [2, 3, 4],
# ]).float()
# target = torch.tensor([
#     [0, 1, 0],
# ]).float()
# target = torch.argmax(target, dim=1)
# entroy = nn.CrossEntropyLoss()
#
# loss = entroy(predicts, target)
#
# predicts = torch.tensor([
#     [1, 2, 3],
# ]).float()
# target = torch.tensor([
#     [0, 0, 1],
# ]).float()
# target = torch.argmax(target, dim=1)
# entroy = nn.CrossEntropyLoss()
#
# loss += entroy(predicts, target)
# print(loss)


# f = open('baby names/female.txt', 'r')
# g = open('baby names/same_len.txt', 'w')
# for line in f.readlines():
#     word = line.strip('\n')
#     if len(word) == 4:
#         print(word, file=g)
#
# f.close()
# g.close()



# import numpy as np
#
# a = np.array([3, 0, 2, 1, 4])
# b = np.argsort(a)[::-1]
# print(a)
# print(b)


import matplotlib.pyplot as plt
import numpy as np

f = open('rnn_res.txt', 'r')

x = list(range(5000))
loss = []
for line in f.readlines():
    loss.append(float(line.split()[4]))

f.close()
plt.title('RNN')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.plot(x, loss)
plt.show()
