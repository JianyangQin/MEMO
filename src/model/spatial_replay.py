import torch
from torch.utils.data import DataLoader
import numpy as np
import networkx as nx
from .linearization import empirical_ntk, functional_net
from tqdm import tqdm


def make_influence_spatial_replay(
        device,
        model,
        task_cur_data,
        task_prev_data,
        task_cur_adj,
        task_prev_adj,
        n_knowledge_nodes=40,
        x_len=12,
        y_len=12,
        time_per_day=288,
        save_path=None,
        load_path=None
):

    knowledge_list = list()

    prev_nodes = np.asarray(range(0, len(task_prev_adj)))

    if load_path is None:
        cross_covs = get_cross_covs(device, model, task_prev_data, task_cur_data[:, prev_nodes],
                              task_prev_adj[prev_nodes][:, prev_nodes],
                              x_len, y_len, time_per_day)
    else:
        cross_covs = np.load(load_path)['cross_covs']
        cross_covs = torch.from_numpy(cross_covs)

    if save_path is not None:
        np.savez(save_path, cross_covs=cross_covs.cpu().numpy())

    graph_node_from_edge = _get_graph_node_from_edge(task_prev_adj)

    memorise_list, forget_list, replay_list = select_covs(
        n_knowledge_nodes, cross_covs, graph_node_from_edge, prev_nodes
    )
    knowledge_list.extend(memorise_list)
    knowledge_list.extend(forget_list)

    return knowledge_list, memorise_list, forget_list, replay_list


def _get_graph_node_from_edge(adj):
    edge_list = list(nx.from_numpy_matrix(adj.numpy()).edges)
    graph_node_from_edge = set()
    for (u, v) in edge_list:
        graph_node_from_edge.add(u)
        graph_node_from_edge.add(v)
    graph_node_from_edge = list(graph_node_from_edge)
    return graph_node_from_edge


def select_covs(n_knowledge_nodes, cross_covs, graph_nodes, prev_nodes):
    n_knowledge_nodes = int(n_knowledge_nodes / 4)

    covs = cross_covs + cross_covs.T
    covs = covs.sum(axis=-1)[graph_nodes]

    _, sort_idx = torch.sort(covs, descending=True)
    sort_nodes = torch.IntTensor(graph_nodes).to(sort_idx.device)[sort_idx]

    mid = int(len(sort_nodes) / 2.)

    mid_half_list = sort_nodes[mid - n_knowledge_nodes:mid + n_knowledge_nodes]
    high_half_list = sort_nodes[:n_knowledge_nodes * 2]
    low_half_list = sort_nodes[-n_knowledge_nodes * 2:]

    memorise_list = torch.cat([high_half_list[:n_knowledge_nodes], low_half_list[-n_knowledge_nodes:]], dim=0)
    forget_list = mid_half_list

    replay_list = torch.cat([high_half_list[:n_knowledge_nodes], mid_half_list, low_half_list[-n_knowledge_nodes:]],
                            dim=0)

    memorise_list = list(memorise_list.detach().cpu().numpy())
    forget_list = list(forget_list.detach().cpu().numpy())
    replay_list = list(replay_list.detach().cpu().numpy())

    memorise_list = list(prev_nodes[memorise_list])
    forget_list = list(prev_nodes[forget_list])
    replay_list = list(prev_nodes[replay_list])

    return memorise_list, forget_list, replay_list


def get_cross_covs(device, model, prev_data, cur_data, adj, x_len, y_len, time_per_day=288, graph_normal=True):
    assert prev_data.shape == cur_data.shape
    node_size = prev_data.shape[-1]

    # daily_prev_data = np.reshape(prev_data[-time_per_day * 7 - 1:-1, :], (-1, x_len, node_size))
    daily_prev_data = np.mean(
        [prev_data[:, :][time_per_day * 7 * i: time_per_day * 7 * (i + 1)] for i in
         range(prev_data.shape[0] // (time_per_day * 7))],
        axis=0
    ).reshape(-1, x_len, node_size)
    daily_prev_data = torch.Tensor(daily_prev_data)
    daily_prev_data = daily_prev_data.permute((0, 2, 1))

    daily_cur_data = np.mean(
        [cur_data[:, :][time_per_day * 7 * i: time_per_day * 7 * (i + 1)] for i in
         range(cur_data.shape[0] // (time_per_day * 7))],
        axis=0
    ).reshape(-1, x_len, node_size)
    daily_cur_data = torch.Tensor(daily_cur_data)
    daily_cur_data = daily_cur_data.permute((0, 2, 1))

    dataset = torch.utils.data.TensorDataset(daily_prev_data, daily_cur_data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    nodes_with_edge = torch.nonzero(adj)
    head_nodes, tail_nodes = nodes_with_edge[:, 0], nodes_with_edge[:, 1]

    node_size = len(nodes_with_edge)

    range_dims_per_node = list(np.arange(0, node_size, 10))
    range_dims_per_node.append(node_size)

    params = {k: v.detach() for k, v in model.named_parameters()}
    node_covs = torch.zeros_like(adj).to(device)

    for batch_idx, ds in enumerate(dataloader):
        batch = ds[0].shape[0]
        for node_idx in tqdm(range(1, len(range_dims_per_node))):
            min_node = range_dims_per_node[node_idx - 1]
            max_node = range_dims_per_node[node_idx]

            hi_nodes, ti_nodes = head_nodes[min_node:max_node], tail_nodes[min_node:max_node]
            select_nodes = torch.unique(torch.cat([hi_nodes, ti_nodes]))

            x1 = ds[0][:, select_nodes, :].to(device)
            x2 = ds[1][:, select_nodes, :].to(device)
            adj_mx = adj[select_nodes][:, select_nodes].to(device)
            mapping = list(range(len(adj_mx)))

            cov = empirical_ntk(
                functional_net,
                model,
                params,
                x1,
                x2,
                adj_mx,
                mapping,
                compute='full'
            )

            cov = torch.einsum('NMNL->NML', cov)
            cov = cov.reshape(-1, len(select_nodes), y_len, len(select_nodes), y_len)
            cov = cov.mean(dim=(0, 2, 4))
            for i, ni in enumerate(select_nodes):
                for j, nj in enumerate(select_nodes):
                    node_covs[ni, nj] = node_covs[ni, nj] + cov[i, j]
    node_covs = node_covs / len(dataloader)
    return node_covs
