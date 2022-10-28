from re import I
import torch
import numpy as np

a=np.array([[1,2],[3,4]])
b=np.array([[2,3],[4,5]])

loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)

input = torch.autograd.Variable(torch.from_numpy(a))
target = torch.autograd.Variable(torch.from_numpy(b))

loss = loss_fn(input.float(), target.float())

batch_size = input.size(0)
num_joints = input.size(1)
x = input.reshape((2,2,-1))
y = target.reshape((2,2,-1))
heatmaps_pred = input.reshape((batch_size, num_joints, -1)).split(1, 1)
heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
loss = 0

print(a)
print(b)
print('\n')

print(x)
print(x.shape)
print('\n')

print(y)
print(y.shape)
print('\n')

print(input)
print(input.shape)
print('\n')

print(target)
print(target.shape)
print('\n')

print(heatmaps_gt)
print([len(a) for a in heatmaps_gt])
print('\n')

print(heatmaps_pred)
print([len(a) for a in heatmaps_pred])
print('\n')
