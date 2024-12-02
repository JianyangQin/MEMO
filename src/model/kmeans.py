import torch
import numpy as np
from tqdm import tqdm

def pairwise_distance(data1, data2=None, device=-1):
	r'''
	using broadcast mechanism to calculate pairwise ecludian distance of data
	the input data is N*M matrix, where M is the dimension
	we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
	then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
	'''
	if data2 is None:
		data2 = data1

	if device!=-1:
		data1, data2 = data1.cuda(device), data2.cuda(device)

	#N*1*M
	A = data1.unsqueeze(dim=1)

	#1*N*M
	B = data2.unsqueeze(dim=0)

	dis = (A-B)**2.0
	#return N*N matrix for pairwise distance
	dis = dis.sum(dim=-1).squeeze()
	return dis

def forgy(X, n_clusters):
    _len = len(X)
    indices = np.random.choice(_len, n_clusters)
    initial_state = X[indices]
    return initial_state


def KMeans_cosine(X, n_clusters, init=None, device=0, niter=300, tol=1e-8):
    X = torch.from_numpy(X).float().cuda(device)

    if init is None:
        initial_state = forgy(X, n_clusters)
    else:
        initial_state = torch.from_numpy(init).float().cuda(device)

    for i in range(niter):
        dis = pairwise_distance(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(n_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze(dim=-1)

            if len(selected) == 0:
                if i == 0:
                    print('reset centroid')
                    x_idx = list(np.random.choice(range(len(X)), 1))
                    initial_state[index] = X[x_idx]
                else:
                    print('centroid error, cluster break in {} iteration'.format(i))
                    return initial_state_pre.cpu().numpy(), choice_cluster_pre.cpu().numpy()
            else:
                selected = torch.index_select(X, 0, selected)
                initial_state[index] = selected.mean(dim=0)

        choice_cluster_pre = choice_cluster.clone()

        center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))

        if center_shift ** 2 < tol:
            print('cluster break in {} iteration'.format(i))
            break

    return initial_state.cpu().numpy(), choice_cluster.cpu().numpy()

def k_cluster(data, n_clusters, init_centroid, len_pred, time_per_day, device):
    # pattern_data = data.reshape(-1, len_pred).transpose(1, 0)

    daily_data = np.mean(
        [data[:, :, 0][time_per_day * i: time_per_day * (i + 1)] for i in range(data.shape[0] // time_per_day)],
        axis=0
    )

    n_nodes = daily_data.shape[-1]
    pattern_data = daily_data.reshape(-1, len_pred, n_nodes).transpose(1, 0, 2)
    pattern_data = pattern_data.reshape(len_pred, -1)

    if init_centroid is not None:
        centroid, label = KMeans_cosine(pattern_data.T, n_clusters, init=init_centroid, device=device)
    else:
        centroid, label = KMeans_cosine(pattern_data.T, n_clusters, device=device)

    return centroid, label


def cosine_sim(device, data, centroid, kc=5):
    n_clusters = centroid.shape[0]

    # 计算每个矩阵的范数
    norm_data = np.linalg.norm(data, axis=-1, keepdims=True)
    norm_cent = np.linalg.norm(centroid, axis=-1, keepdims=True)

    # 计算点积
    dot_product = np.dot(data, centroid.T)

    # 计算余弦相似度
    sim = dot_product / (norm_data * norm_cent.T)

    # 找到第三维中的最大值索引
    sorted_indices = np.argsort(-sim, axis=-1)
    top_n_indices = sorted_indices[..., :kc]

    # 创建一个全零矩阵，与第一、二维相同，以便保留第一、二维
    filter_sim = np.zeros_like(sim)

    # 将最大值索引保留在第一、二维中
    for i in tqdm(range(sim.shape[0])):
        for j in range(sim.shape[1]):
            max_idx = top_n_indices[i, j]
            filter_sim[i, j, max_idx] = sim[i, j, max_idx]

    return filter_sim