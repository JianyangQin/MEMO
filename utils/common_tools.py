import os
import shutil
import networkx as nx
import re
import json
import numpy as np
import logging
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
import gc
import os.path as osp
import torch

from src.model.model import Basic_Model, PECPM, Basic_GCN

def load_best_model(args):
    if (args.load_first_year and args.year <= args.begin_year + 1) or args.train == 0:
        load_path = args.first_year_model_path
        loss = load_path.split("/")[-1].replace(".pkl", "")
    else:
        loss = []
        for filename in os.listdir(osp.join(args.model_path, args.logname + args.time, str(args.year - 1))):
            if filename.endswith('pkl'):
                loss.append(float(filename[0:-4]))
        loss = sorted(loss)
        load_path = osp.join(args.model_path, args.logname + args.time, str(args.year - 1), str(loss[0]) + ".pkl")

    args.logger.info("[*] load from {}".format(load_path))
    state_dict = torch.load(load_path, map_location=args.device)["model_state_dict"]
    if 'tcn2.weight' in state_dict:
        del state_dict['tcn2.weight']
        del state_dict['tcn2.bias']
    if args.strategy == "incremental":
        model = PECPM(args) if args.pattern else Basic_GCN(args)
    else:
        model = Basic_Model(args)
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    return model, loss[0]

def find_positions(src, des):
    positions = []

    for i, element in enumerate(des):
        if element in src:
            positions.append(i)

    return positions

def create_subgraph(subgraph, subgraph_edge_index, graph_normal=True):
    graph = nx.Graph()
    graph.add_nodes_from(range(subgraph.size(0)))
    graph.add_edges_from(subgraph_edge_index.numpy().T)
    adj = nx.to_numpy_array(graph)
    if graph_normal:
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
    return graph, adj

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def graph_matrix_reader(file):
    df = pd.read_csv(file, header=None, index_col=None)
    return np.asarray(df.values)

def dict_add(d, key, add):
    if key in d:
        d[key] += add
    else:
        d[key] = add

def check_attr(params, attr, default):
    if attr not in params:
        params[attr] = default
        return False
    return True

def obj_dic(d):
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(top, i, obj_dic(j))
        elif isinstance(j, seqs):
            setattr(top, i,
                type(j)(obj_dic(sj) if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(top, i, j)
    return top

def load_fea(file_path):
    X = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            items = line.split()
            if len(items) < 1:
                continue
            X.append([float(item) for item in items])
    return np.array(X)


def symlink(src, dst):
    try:
        os.symlink(src, dst)
    except OSError:
        os.remove(dst)
        os.symlink(src, dst)


def load_json_file(file_path):
    with open(file_path, "r") as f:
        s = f.read()
        s = re.sub('\s',"", s)
    return json.loads(s)

def get_time_str():
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")

def append_to_file(file_path, s):
    with open(file_path, "a") as f:
        f.write(s)

def rmtree(path):
    if os.path.exists(path) == True:
        shutil.rmtree(path)
        return True
    return False

def get_logger(log_filename=None, module_name=__name__, level=logging.INFO):
    # select handler
    if log_filename is None:
        handler = logging.StreamHandler()
    elif type(log_filename) is str:
        handler = logging.FileHandler(log_filename, 'w')
    else:
        raise ValueError("log_filename invalid!")

    # build logger
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    handler.setLevel(level)
    formatter = logging.Formatter(('%(asctime)s %(filename)s' \
                    '[line:%(lineno)d] %(levelname)s %(message)s'))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def load_ground_truth(file_path):
    lst = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            lst.append([int(i) for i in items])
    lst.sort()
    return [i[1] for i in lst]

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        res = func(*args, **kwargs)
        end_time = datetime.now()
        print((end_time - start_time).seconds)
        return res
    return wrapper



def module_decorator(func):
    def wrapper(*args, **kwargs):
        print("[+] Start %s ..." % (kwargs["mdl_name"], ))
        kwargs["info"]["log"].info("Start Module %s" % (kwargs["mdl_name"], ))
        start_time = datetime.now()
        res = func(*args, **kwargs)
        end_time = datetime.now()
        gc.collect()
        print("[+] Finished!\n[+] During Time: %.2f\n"  % (end_time - start_time).seconds)
        kwargs["info"]["log"].info(
                "[+] Finished!\n[+] During Time: %.2f\n" % (end_time - start_time).seconds)
        res["Duration"] = (end_time - start_time).seconds
        kwargs["info"]["log"].info("Module Results: " + str(res))
        print("[+] Module Results: " + str(res))
        kwargs["info"]["log"].info("[+] Module Results: " + str(res))
        return res
    return wrapper

def load_multilabel_ground_truth(file_path):
    lst = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            lst.append([int(i) for i in items])
    lst.sort()
    lst = [i[1:] for i in lst]
    mlb = MultiLabelBinarizer()
    return mlb.fit_transform(lst)

def load_onehot_ground_truth(file_path):
    lst = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            lst.append([int(i) for i in items])
    lst.sort()
    return np.array([i[1:] for i in lst], dtype=int)
