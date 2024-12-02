import torch
import numpy as np
from torch_geometric.data import DataLoader
from utils.common_tools import create_subgraph, load_best_model
from torch_geometric.utils import k_hop_subgraph
from src.trafficDataset import TrafficDataset
from src.model.model import PECPM, Basic_GCN
from src.model.linearization import bnn_linearized_predictive


import networkx as nx
from tqdm import tqdm

def make_influence_temporal_replay(inputs, args, save_path=None, load_path=None):
    # Dataset Definition
    if args.strategy == 'incremental':
        if load_path is None:
            if args.year > args.begin_year:
                graph, adj = create_subgraph(args.subgraph, args.subgraph_edge_index)

                node_list = list(np.random.choice(range(len(adj)), 10))
                graph_node_from_edge = set()
                for (u, v) in list(graph.edges):
                    graph_node_from_edge.add(u)
                    graph_node_from_edge.add(v)
                node_list = list(set(node_list) & graph_node_from_edge)
                subgraph, subgraph_edge_index, mapping, _ = k_hop_subgraph(node_list, num_hops=args.num_hops,
                                                                           edge_index=args.subgraph_edge_index,
                                                                           relabel_nodes=True)

                x_train = inputs["train_x"][:, :, args.subgraph.numpy()]
                y_train = inputs["train_y"][:, :, args.subgraph.numpy()]

                temporal_replay_rate = args.temporal_replay_rate
            else:
                adj = args.adj.detach().cpu().numpy()
                graph = nx.from_numpy_matrix(adj)
                graph_edge_index = torch.LongTensor(np.array(list(graph.edges)).T)

                node_list = list(np.random.choice(range(len(adj)), 10))
                graph_node_from_edge = set()
                for (u, v) in list(graph.edges):
                    graph_node_from_edge.add(u)
                    graph_node_from_edge.add(v)
                node_list = list(set(node_list) & graph_node_from_edge)
                subgraph, subgraph_edge_index, mapping, _ = k_hop_subgraph(node_list, num_hops=args.num_hops,
                                                                           edge_index=graph_edge_index,
                                                                           relabel_nodes=True)

                x_train = inputs["train_x"]
                y_train = inputs["train_y"]

                temporal_replay_rate = 1.0

            _, sub_adj = create_subgraph(subgraph, subgraph_edge_index)
            sub_adj = torch.from_numpy(sub_adj).to(torch.float).to(args.device)

            x_train = x_train[:, :, subgraph.numpy()]
            y_train = y_train[:, :, subgraph.numpy()]

            train_loader = DataLoader(
                TrafficDataset("", "", x=x_train, y=y_train, k=x_train, edge_index="", mode="subgraph"), \
                batch_size=args.objective_samples, shuffle=True, pin_memory=False, num_workers=0)


            # Model Definition
            if args.init == True and args.year > args.begin_year:
                model, _ = load_best_model(args)
            else:
                model = PECPM(args).to(args.device) if args.pattern else Basic_GCN(args).to(args.device)

            model.eval()

            n_nodes = len(sub_adj)

            params = {k: v.detach() for k, v in model.named_parameters()}
            covs = []

            for data in tqdm(train_loader):
                inp = data[0].to(args.device)
                mapping = list(range(len(sub_adj)))

                _, cov = bnn_linearized_predictive(
                    model,
                    params,
                    inp,
                    sub_adj,
                    mapping,
                    full_ntk=True,
                    for_loop=False,
                    identity_cov=False
                )

                cov = torch.einsum('NMNL->NML', cov)
                cov = cov.reshape(-1, n_nodes, args.y_len, n_nodes, args.y_len)
                cov = cov[:, mapping][:, :, :, mapping, :]
                cov = cov.mean(dim=(1, 2, 3, 4))

                covs.append(cov)

            covs = torch.cat(covs, dim=0)
        else:
            covs = np.load(load_path)['covs']
            covs = torch.from_numpy(covs)
            temporal_replay_rate = args.temporal_replay_rate

        if save_path is not None:
            np.savez(save_path, covs=covs.cpu().numpy())


        _, sort_idx = torch.sort(covs.view(-1), descending=True)
        ntk_samples = int(len(sort_idx) * temporal_replay_rate / 2)
        sample_idx = (np.concatenate([sort_idx[:ntk_samples].cpu().numpy(),
                                      sort_idx[-ntk_samples:].cpu().numpy()]))


        outputs = {}
        outputs['train_x'] = inputs['train_x'][sample_idx]
        outputs['train_y'] = inputs['train_y'][sample_idx]
        outputs['val_x'] = inputs['val_x']
        outputs['val_y'] = inputs['val_y']
        outputs['test_x'] = inputs['test_x']
        outputs['test_y'] = inputs['test_y']
    else:
        outputs = inputs

    return outputs