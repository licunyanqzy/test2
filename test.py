import torch.nn as nn
import torch
import math
from multiprocessing.dummy import Pool as ThreadPool
import time
import numpy as np


# def solution(height):
#     l = len(height)
#     outputs = []
#     for i in range(1, l):
#         sub = height[i:]
#         head = [height[i-1]] * (l-i)
#         minimum = list(map(lambda x, y: min(x, y), sub, head))
#         index = list(range(1, l-i+1))
#         multiple = list(map(lambda x, y: x * y, minimum, index))
#         outputs.append(max(multiple))
#     return max(outputs)
#
#
# height = [1, 1]     # [1, 8, 6, 2, 5, 4, 8, 3, 7]
# output = solution(height)
# print(output)


# def func(i):
#     t = time.asctime(time.localtime(time.time()))
#     print(t)
#     return i, t
#
#
# inputs = [np.arange(1,3), np.arange(3,5)]
# pool = ThreadPool()
# results = pool.map(func, inputs)
# pool.close()
# pool.join()
#
# print(results)


def kmeans(x, ncluster, niter=10):
    '''
    x : torch.tensor(data_num,data_dim)
    ncluster : The number of clustering for data_num
    niter : Number of iterations for kmeans
    '''
    N, D = x.size()
    c = x[torch.randperm(N)[:ncluster]] # init clusters at random
    for i in range(niter):
        # assign all pixels to the closest codebook element
        # .argmin(1) : 按列取最小值的下标,下面这行的意思是将x.size(0)个数据点归类到random选出的ncluster类
        a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
        # move each codebook element to be the mean of the pixels that assigned to it
        # 计算每一类的迭代中心，然后重新把第一轮随机选出的聚类中心移到这一类的中心处
        c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
        # re-assign any poorly positioned codebook elements
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        print('done step %d/%d, re-initialized %d dead clusters' % (i+1, niter, ndead))
        c[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters
    return c


data = torch.rand(8, 2)
group = kmeans(data, 3)
print(group)


