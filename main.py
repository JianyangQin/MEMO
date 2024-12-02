import warnings
warnings.filterwarnings('ignore')

import sys, argparse, random, os
sys.path.append("src/")
import numpy as np
import logging
from datetime import datetime
import os.path as osp
import networkx as nx

import torch
import torch.nn.functional as func
from torch import optim
from torch.utils.data import DataLoader
from torch_geometric.utils import k_hop_subgraph

from utils import common_tools as ct
from utils.common_tools import load_best_model, find_positions, create_subgraph
from utils.my_math import masked_mae_np, masked_mape_np, masked_mse_np
from utils.data_convert import generate_samples
from src.trafficDataset import TrafficDataset
from src.model.model import Basic_Model, PECPM, Basic_GCN
from src.model.objective import consolidation_loss, pred_cls_loss, pred_loss
from src.model.prior import CLPrior
from src.model.spatial_replay import make_influence_spatial_replay
from src.model.temporal_replay import make_influence_temporal_replay
from src.model import replay
from src.model.kmeans import k_cluster, cosine_sim
from copy import deepcopy


result = {}
pin_memory = False


def update(src, tmp):
    for key in tmp:
        if key!= "gpuid":
            src[key] = tmp[key]

def init(args):    
    conf_path = osp.join(args.conf)
    info = ct.load_json_file(conf_path)
    info["time"] = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    update(vars(args), info)
    vars(args)["path"] = osp.join(args.model_path, args.logname+args.time)
    ct.mkdirs(args.path)
    del info


def init_log(args):
    log_dir, log_filename = args.path, args.logname
    logger = logging.getLogger(__name__)
    ct.mkdirs(log_dir)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(osp.join(log_dir, log_filename+".log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("logger name:%s", osp.join(log_dir, log_filename+".log"))
    vars(args)["logger"] = logger
    return logger


def seed_set(seed=0):
    max_seed = (1 << 32) - 1
    random.seed(seed)
    np.random.seed(random.randint(0, max_seed))
    torch.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed_all(random.randint(0, max_seed))
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True


def train(inputs, prev_centroid, args):
    # Model Setting
    global result
    path = osp.join(args.path, str(args.year))
    ct.mkdirs(path)

    if args.strategy == "incremental" and args.pattern: lossfunc = pred_cls_loss
    else: lossfunc = pred_loss

    # Dataset Cluster
    centroid = None
    if args.strategy == 'incremental':
        train_inputs = inputs["train_x"][:, :, args.subgraph.numpy()].transpose(0, 2, 1) if args.year > args.begin_year else inputs["train_x"].transpose(0, 2, 1)
        val_inputs = inputs["val_x"][:, :, args.subgraph.numpy()].transpose(0, 2, 1) if args.year > args.begin_year else inputs["val_x"].transpose(0, 2, 1)
        args.logger.info("clustering samples.....")
        centroid, _ = k_cluster(
            data=train_inputs,
            n_clusters=args.n_clusters,
            init_centroid=prev_centroid if args.keep_centroid is True else None,
            len_pred=args.x_len,
            time_per_day=args.time_per_day,
            device=args.device
        )
        args.logger.info("generating train sample labels.....")
        k_train = cosine_sim(args.device, train_inputs, centroid, args.kc).transpose(0, 2, 1)
        args.logger.info("generating val sample labels.....")
        k_val = cosine_sim(args.device, val_inputs, centroid, args.kc).transpose(0, 2, 1)
        args.logger.info("clustering and generating labels finish!")

    # Dataset Definition
    if args.strategy == 'incremental':
        x_train = inputs["train_x"][:, :, args.subgraph.numpy()] if args.year > args.begin_year else inputs["train_x"]
        y_train = inputs["train_y"][:, :, args.subgraph.numpy()] if args.year > args.begin_year else inputs["train_y"]
        x_val = inputs["val_x"][:, :, args.subgraph.numpy()] if args.year > args.begin_year else inputs["val_x"]
        y_val = inputs["val_y"][:, :, args.subgraph.numpy()] if args.year > args.begin_year else inputs["val_y"]

        if args.shuffle:
            train_loader = DataLoader(TrafficDataset("", "", x=x_train, y=y_train, k=k_train, edge_index="", mode="subgraph"), \
                batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory, num_workers=args.n_work)
        else:
            train_loader = DataLoader(TrafficDataset("", "", x=x_train, y=y_train, k=k_train, edge_index="", mode="subgraph"), \
                batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=args.n_work)
        val_loader = DataLoader(TrafficDataset("", "", x=x_val, y=y_val, k=k_val, edge_index="", mode="subgraph"), \
            batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=args.n_work)

        if args.year > args.begin_year:
            graph, adj = create_subgraph(args.subgraph, args.subgraph_edge_index, args.graph_normal)
            vars(args)["sub_adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)

            if args.ntk:
                edge_idx = torch.LongTensor(np.array(list(graph.edges)).T)

                origin_mapping_idx = list(vars(args)["mapping"].cpu().numpy())

                memorise_nodes = list(set(args.memorise_nodes) & set(args.node_list))
                forget_nodes = list(set(args.forget_nodes) & set(args.node_list))
                replay_nodes = list(set(args.replay_nodes) & set(args.node_list))

                memorise_idx = find_positions(memorise_nodes, list(args.subgraph.numpy()))
                forget_idx = find_positions(forget_nodes, list(args.subgraph.numpy()))
                replay_idx = find_positions(replay_nodes, list(args.subgraph.numpy()))

                args.logger.info("[*] memorise Nodes: " + str(args.memorise_nodes))
                args.logger.info("[*] forget Nodes: " + str(args.forget_nodes))

                knowledge_idx = memorise_idx + forget_idx

                mapping_idx = set(origin_mapping_idx) - set(memorise_idx) - set(forget_idx)
                mapping_idx = list(mapping_idx | set(replay_idx))
                vars(args)["mapping"] = mapping_idx
                args.logger.info("[*] Replay Nodes: " + str(args.mapping))

                knowledge_subgraph, knowledge_subgraph_edge_index, knowledge_mapping, _ = k_hop_subgraph(
                    knowledge_idx, num_hops=args.num_hops, edge_index=edge_idx, relabel_nodes=True
                )

                memorise_mapping = find_positions(memorise_idx, list(knowledge_subgraph.numpy()))
                forget_mapping = find_positions(forget_idx, list(knowledge_subgraph.numpy()))

                _, knowledge_adj = create_subgraph(knowledge_subgraph, knowledge_subgraph_edge_index, args.graph_normal)
        else:
            vars(args)["sub_adj"] = vars(args)["adj"]
    else:
        if args.shuffle:
            train_loader = DataLoader(TrafficDataset(inputs, "train"), batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory, num_workers=args.n_work)
        else:
            train_loader = DataLoader(TrafficDataset(inputs, "train"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=args.n_work)
        val_loader = DataLoader(TrafficDataset(inputs, "val"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=args.n_work)
        vars(args)["sub_adj"] = vars(args)["adj"]
    test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=args.n_work)

    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

    # Model Definition
    if args.init == True and args.year > args.begin_year:
        model, _ = load_best_model(args)
        prior_model = deepcopy(model)
        prior_fn = args.cl_prior.make_prior_fn(prior_model, args.full_ntk)
    else:
        if args.strategy == "incremental":
            model = PECPM(args).to(args.device) if args.pattern else Basic_GCN(args).to(args.device)
        else:
            model = Basic_Model(args).to(args.device)
        prior_model = deepcopy(model)
        prior_fn = args.cl_prior.make_prior_fn(prior_model, args.full_ntk, "fixed")

    # Model Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    args.logger.info("[*] Year " + str(args.year) + " Training start")

    lowest_validation_loss = 1e7
    counter = 0
    patience = args.patience
    model.train()
    use_time = []
    for epoch in range(args.epoch):
        training_loss = 0.0
        start_time = datetime.now()
        
        # Train Model
        cn = 0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data[0].shape))
            data_x = data[0].to(args.device)
            data_y = data[1].to(args.device)
            data_k = data[2].to(args.device)

            optimizer.zero_grad()
            if args.strategy == 'incremental' and args.pattern:
                pred, pattern = model.predict(data_x, args.sub_adj, args.mapping)
            else:
                pred = model.predict(data_x, args.sub_adj, args.mapping)
          
            if args.strategy == "incremental" and args.year > args.begin_year:
                pred = pred.reshape(-1, len(args.mapping), args.y_len)
                data_y = data_y[:, args.mapping, :]
                data_k = data_k[:, args.mapping, :]

            if args.strategy == "incremental":
                if args.pattern:
                    loss = lossfunc(data_y, pred, data_k, pattern)
                else:
                    loss = lossfunc(data_y, pred)
                if args.year > args.begin_year:
                    context_points = data_x
                    n_samples = args.objective_samples if (args.objective_samples is not None) and (args.objective_samples < len(context_points)) else len(context_points)
                    sample_idx = list(random.sample(range(len(context_points)), n_samples))
                    context_points = context_points[sample_idx]
                    context_points = context_points[:, knowledge_subgraph]
                    context_adjs = torch.Tensor(knowledge_adj).type(torch.float32).to(args.device)
                    loss += consolidation_loss(
                      model,
                      prior_fn,
                      context_points,
                      context_adjs,
                      memorise_mapping,
                      forget_mapping,
                      args.forget_weight,
                      args.device,
                      args.full_ntk,
                      args.y_len
                    )
            else:
                loss = lossfunc(data.y, pred)
            training_loss += float(loss)
            loss.backward()
            optimizer.step()
            
            cn += 1

        if epoch == 0:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time += (datetime.now() - start_time).total_seconds()
        use_time.append((datetime.now() - start_time).total_seconds())
        training_loss = training_loss/cn 
 
        # Validate Model
        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data_x = data[0].to(args.device)
                data_y = data[1].to(args.device)
                if args.strategy == 'incremental' and args.pattern:
                    pred, pattern = model.predict(data_x, args.sub_adj, args.mapping)
                else:
                    pred = model.predict(data_x, args.sub_adj, args.mapping)
                pred = pred.reshape(-1, len(args.mapping), args.y_len)
                if args.strategy == "incremental" and args.year > args.begin_year:
                    data_y = data_y[:, args.mapping, :]
                loss = masked_mae_np(data_y.cpu().data.numpy(), pred.cpu().data.numpy(), 0)
                validation_loss += float(loss)
                cn += 1
        validation_loss = float(validation_loss/cn)
        

        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")

        # Early Stop
        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            torch.save({'model_state_dict': model.state_dict()}, osp.join(path, str(round(validation_loss,4))+".pkl"))
        else:
            counter += 1
            if counter > patience:
                break

    best_model_path = osp.join(path, str(lowest_validation_loss)+".pkl")
    if args.strategy == "incremental":
        best_model = PECPM(args) if args.pattern else Basic_GCN(args)
    else:
        best_model = Basic_Model(args)
    best_model.load_state_dict(torch.load(best_model_path, args.device)["model_state_dict"])
    best_model = best_model.to(args.device)
    
    # Test Model
    test_model(best_model, args, test_loader)

    result[args.year] = {"total_time": total_time, "average_time": sum(use_time)/len(use_time), "epoch_num": epoch+1}
    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))

    return centroid


def test_model(model, args, testset):
    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0
    with torch.no_grad():
        cn = 0
        mapping = list(range(len(args.adj)))
        for data in testset:
            data_x = data[0].to(args.device)
            if args.strategy == 'incremental' and args.pattern:
                pred, pattern = model.predict(data_x, args.adj, mapping)
            else:
                pred = model.predict(data_x, args.adj, mapping)
            true = data[1].to(args.device)
            if args.strategy == "incremental":
                pred = pred.reshape(-1, len(args.adj), args.y_len)
            loss += func.mse_loss(true, pred, reduction="mean")
            pred_.append(pred.cpu().data.numpy())
            truth_.append(true.cpu().data.numpy())
            cn += 1
        loss = loss/cn
        args.logger.info("[*] loss:{:.4f}".format(loss))
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)
        metric(truth_, pred_, args)
    return truth_, pred_


def metric(ground_truth, prediction, args):
    global result
    pred_time = args.pred_time
    args.logger.info("[*] year {}, testing".format(args.year))
    for i in pred_time:
        mae = masked_mae_np(ground_truth[..., :i], prediction[..., :i], 0)
        rmse = masked_mse_np(ground_truth[..., :i], prediction[..., :i], 0) ** 0.5
        mape = masked_mape_np(ground_truth[..., :i], prediction[..., :i], 10.)
        args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i,mae,rmse,mape))
        result[i]["mae"][args.year] = mae
        result[i]["mape"][args.year] = mape
        result[i]["rmse"][args.year] = rmse
    return mae


def main(args):
    logger = init_log(args)
    logger.info("params : %s", vars(args))
    ct.mkdirs(args.save_data_path)

    init_centroid = None

    cl_prior = CLPrior(
        prior_type="bnn_induced",
        output_size=12,
        full_ntk=True,
        prior_mean=0.5,
        prior_cov=0.01
    )
    vars(args)["cl_prior"] = cl_prior
    vars(args)["year_graph_size"] = [0,]


    for year in range(args.begin_year, args.end_year+1):
        vars(args)["forget_weight"] = args.forget_weights[year - args.begin_year]

        # Load Data 
        graph = nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"])
        vars(args)["graph_size"] = graph.number_of_nodes()
        vars(args)["year"] = year
        vars(args)["year_graph_size"].append(graph.number_of_nodes())
        inputs = generate_samples(31, osp.join(args.save_data_path, str(year)+'_30day'), np.load(osp.join(args.raw_data_path, str(year)+".npz")),
                                  graph, val_test_mix=False, pre_scalar=args.pre_scalar, x_len=args.x_len, y_len=args.y_len) \
            if args.data_process else np.load(osp.join(args.save_data_path, str(year)+"_30day.npz"), allow_pickle=True)

        args.logger.info("[*] Year {} load from {}_30day.npz".format(args.year, osp.join(args.save_data_path, str(year)))) 

        adj = np.load(osp.join(args.graph_path, str(args.year)+"_adj.npz"))["x"]
        if args.graph_normal:
            adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)

        if year == args.begin_year and args.load_first_year:
            # Skip the first year, model has been trained and retrain is not needed
            model, _ = load_best_model(args)
            test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=args.n_work)
            test_model(model, args, test_loader)
            continue

        
        if year > args.begin_year and args.strategy == "incremental":
            # Load the best model
            model, _ = load_best_model(args)
            
            node_list = list()
            # Obtain increase nodes
            if args.increase:
                cur_node_size = np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"].shape[0]
                pre_node_size = np.load(osp.join(args.graph_path, str(year-1)+"_adj.npz"))["x"].shape[0]
                node_list.extend(list(range(pre_node_size, cur_node_size)))
            
            # Obtain NTK nodes for memorising & forgetting
            if args.ntk:
                args.logger.info("[*] detect ntk influence strategy, full ntk: {}".format(args.full_ntk))
                pre_data = np.load(osp.join(args.raw_data_path, str(year - 1) + ".npz"))["x"]
                cur_data = np.load(osp.join(args.raw_data_path, str(year) + ".npz"))["x"]
                pre_graph = np.load(osp.join(args.graph_path, str(year - 1) + "_adj.npz"))["x"]
                cur_graph = np.load(osp.join(args.graph_path, str(year) + "_adj.npz"))["x"]
                if args.graph_normal:
                    pre_graph = torch.from_numpy(pre_graph / (np.sum(pre_graph, 1, keepdims=True) + 1e-6)).type(torch.float32)
                    cur_graph = torch.from_numpy(cur_graph / (np.sum(cur_graph, 1, keepdims=True) + 1e-6)).type(torch.float32)
                else:
                    pre_graph = torch.from_numpy(pre_graph).type(torch.float32)
                    cur_graph = torch.from_numpy(cur_graph).type(torch.float32)

                if args.spatial_replay_rate > 1.:
                    vars(args)["topk"] = int(args.spatial_replay_rate)
                else:
                    vars(args)["topk"] = int(args.spatial_replay_rate * args.graph_size)

                spatial_replay_save_path = osp.join(args.path, str(year)+'_spatial_replay.npz')
                spatial_replay_load_path = args.spatial_replay_load_path if args.spatial_replay_load_path and (args.load_first_year==1) and (year == args.begin_year + 1) else None
                knowledge_node_list, memorise_node_list, forget_node_list, replay_node_list = make_influence_spatial_replay(
                    device, model, cur_data, pre_data, cur_graph, pre_graph,
                    args.topk, args.x_len, args.y_len, args.time_per_day,
                    spatial_replay_save_path, spatial_replay_load_path
                )
                node_list.extend(list(set(knowledge_node_list) | set(replay_node_list)))
                vars(args)["memorise_nodes"] = memorise_node_list
                vars(args)["forget_nodes"] = forget_node_list
                vars(args)["replay_nodes"] = replay_node_list

            # Obtain sample nodes
            if args.replay:
                vars(args)["replay_num_samples"] = int(args.replay_rate * args.graph_size) #int(0.2*args.graph_size)- len(node_list)
                args.logger.info("[*] replay node number {}".format(args.replay_num_samples))
                replay_node_list = replay.replay_node_selection(args, inputs, model)
                node_list.extend(list(replay_node_list))

            node_list = list(set(node_list))
            if len(node_list) > int(args.increase_rate * args.graph_size):
                node_list = random.sample(node_list, int(args.increase_rate * args.graph_size))

            n_memorising_nodes = len(find_positions(memorise_node_list, node_list))
            if n_memorising_nodes < args.min_spatial_replay_nodes:
                args.logger.info("select {} memorising nodes are not enough, adding memorising nodes".format(n_memorising_nodes))
                node_list = list(set(node_list) | set(memorise_node_list[:int(args.min_spatial_replay_nodes/2)]) | set(memorise_node_list[-int(args.min_spatial_replay_nodes/2):]))
            n_forgetting_nodes = len(find_positions(forget_node_list, node_list))
            if n_forgetting_nodes < args.min_spatial_replay_nodes:
                args.logger.info("select {} forgetting nodes are not enough, adding forgetting nodes".format(n_memorising_nodes))
                node_list = list(set(node_list) | set(forget_node_list[:int(args.min_spatial_replay_nodes/2)]) | set(forget_node_list[-int(args.min_spatial_replay_nodes/2):]))
            
            # Obtain subgraph of node list
            cur_graph = torch.LongTensor(np.array(list(nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)).T)
            edge_list = list(nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"]).edges)
            graph_node_from_edge = set()
            for (u,v) in edge_list:
                graph_node_from_edge.add(u)
                graph_node_from_edge.add(v)
            node_list = list(set(node_list) & graph_node_from_edge)
           
            if len(node_list) != 0 :
                subgraph, subgraph_edge_index, mapping, _ = k_hop_subgraph(node_list, num_hops=args.num_hops, edge_index=cur_graph, relabel_nodes=True)
                vars(args)["subgraph"] = subgraph
                vars(args)["subgraph_edge_index"] = subgraph_edge_index
                vars(args)["mapping"] = mapping
            logger.info("number of increase nodes:{}, nodes after {} hop:{}, total nodes this year {}".format\
                        (len(node_list), args.num_hops, args.subgraph.size(), args.graph_size))
            vars(args)["node_list"] = np.asarray(node_list)
        else:
            args.mapping = list(range(len(adj)))


        # Skip the year when no nodes needed to be trained incrementally
        if args.strategy != "retrain" and year > args.begin_year and len(args.node_list) == 0:
            model, loss = load_best_model(args)
            ct.mkdirs(osp.join(args.model_path, args.logname+args.time, str(args.year)))
            torch.save({'model_state_dict': model.state_dict()}, osp.join(args.model_path, args.logname+args.time, str(args.year), loss+".pkl"))
            test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=args.n_work)
            test_model(model, args, test_loader)
            logger.warning("[*] No increasing nodes at year " + str(args.year) + ", store model of the last year.")
            continue

        if args.train:
            if args.strategy == 'incremental' and args.ntk:
                args.logger.info("sampling subset series.....")
                temporal_replay_save_path = osp.join(args.path, str(year)+'_temporal_replay.npz')
                temporal_replay_load_path = args.temporal_replay_load_path if args.temporal_replay_load_path and (args.load_first_year == 1) and (year == args.begin_year + 1) else None
                temporal_replay_inputs = make_influence_temporal_replay(inputs, args, temporal_replay_save_path, temporal_replay_load_path)
                args.logger.info("sampling finish!")
                init_centroid = train(temporal_replay_inputs, init_centroid, args)
            else:
                init_centroid = train(inputs, init_centroid, args)
        else:
            if args.auto_test:
                model, _ = load_best_model(args)
                test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=args.n_work)
                test_model(model, args, test_loader)

    for i in args.pred_time:
        for j in ['mae', 'rmse', 'mape']:
            info = ""
            for year in range(args.begin_year, args.end_year + 1):
                if i in result:
                    if j in result[i]:
                        if year in result[i][j]:
                            info += "{:.2f}\t".format(result[i][j][year])
            logger.info("{}\t{}\t".format(i, j) + info)

    for year in range(args.begin_year, args.end_year + 1):
        if year in result:
            info = "year\t{}\ttotal_time\t{}\taverage_time\t{}\tepoch\t{}".format(year, result[year]["total_time"], result[year]["average_time"], result[year]['epoch_num'])
            logger.info(info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("--conf", type=str, default="conf_uctb/memo.json")
    parser.add_argument("--paral", type = int, default = 0)
    parser.add_argument("--gpuid", type = int, default = 0)
    parser.add_argument("--logname", type = str, default = "info")
    parser.add_argument("--load_first_year", type = int, default = 0, help="0: training first year, 1: load from model path of first year")
    parser.add_argument("--first_year_model_path", type = str, default = "res/district3F11T17/TrafficStream2021-05-09-11:56:33.516033/2011/27.4437.pkl", help='specify a pretrained model root')
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    init(args)

    if args.y_len == 12:
        result = {3: {"mae": {}, "mape": {}, "rmse": {}}, 6: {"mae": {}, "mape": {}, "rmse": {}}, 12: {"mae": {}, "mape": {}, "rmse": {}}}
    elif args.y_len == 3:
        result = {1: {"mae": {}, "mape": {}, "rmse": {}}, 2: {"mae": {}, "mape": {}, "rmse": {}}, 3: {"mae": {}, "mape": {}, "rmse": {}}}
    elif args.y_len == 1:
        result = {1: {"mae": {}, "mape": {}, "rmse": {}}}
    else:
        raise ValueError("definition of result is wrong!")
    seed_set(args.seed)

    device = torch.device("cuda:{}".format(args.gpuid)) if torch.cuda.is_available() and args.gpuid != -1 else "cpu"
    vars(args)["device"] = device
    main(args)