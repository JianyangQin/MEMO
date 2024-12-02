import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
from .common_tools import mkdirs
from tqdm import tqdm

def plot(true_y, pred_y, args):
    max_step = 360
    path = osp.join(args.path, str(args.year), 'results')
    mkdirs(path)
    # n_nodes = true_y.shape[1] if true_y.shape[1] < 10 else 10
    n_nodes = true_y.shape[1]
    for i in range(n_nodes):
        filename = os.path.join(path, 'pred_{}point.png'.format(i))
        time_step = np.arange(0, max_step, 1)
        fig = plt.figure(figsize=(18, 6))
        ax = fig.add_axes([0.15, 0.3, 0.82, 0.5])
        ax.tick_params(labelsize=56)
        ax.plot(time_step, true_y[:max_step, i, 0], color="blue", linewidth=3, label='ground-truth')
        ax.plot(time_step, pred_y[:max_step, i, 0], color="red", linewidth=3, label='prediction')
        ax.legend(fontsize=16, loc='upper right')  # 自动检测要在图例中显示的元素，并且显示
        ax.set_xlabel('tiemslot', fontsize=64)
        ax.set_ylabel('flow', fontsize=64)
        plt.title('Base', fontsize=72)
        plt.savefig(filename, bbox_inches='tight')
        # plt.show()
        
def plot_context(y, args):
    path = osp.join(args.path, str(args.year), 'context')
    mkdirs(path)
    y = y['train_x']
    n_nodes = y.shape[1]
    true_y = y.reshape(-1, n_nodes)
    max_step = len(true_y)
    print("plot context nodes......")
    for i in tqdm(range(n_nodes)):
        filename = os.path.join(path, 'pred_{}point.png'.format(i))
        time_step = np.arange(0, max_step, 1)
        fig = plt.figure(figsize=(18, 6))
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
        ax.plot(time_step, true_y[:max_step, i], color="blue", linewidth=1)
        plt.savefig(filename, bbox_inches='tight')
        # plt.show()

def plot_coreset(y, customary_list, peculiar_list, args):
    path = osp.join(args.path, str(args.year), 'coreset')
    mkdirs(path)
    true_y = y['train_x']
    n_nodes = true_y.shape[-1]
    max_step = len(true_y)
    print("plot coreset nodes......")
    for i in tqdm(range(n_nodes)):
        filename = os.path.join(path, 'pred_{}point.png'.format(i))
        time_step = np.arange(0, max_step, 1)
        fig = plt.figure(figsize=(18, 6))
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
        if i in customary_list:
            ax.plot(time_step, true_y[:max_step, 0, i], color="red", linewidth=1)
        elif i in peculiar_list:
            ax.plot(time_step, true_y[:max_step, 0, i], color="magenta", linewidth=1)
        else:
            ax.plot(time_step, true_y[:max_step, 0, i], color="black", linewidth=1)
        plt.savefig(filename, bbox_inches='tight')
        # plt.show()
        
# def plot_coreset(y, conflict_list, stable_list, replay_list, args):
#     path = osp.join(args.path, str(args.year), 'coreset')
#     mkdirs(path)
#     true_y = y['train_x']
#     n_nodes = true_y.shape[-1]
#     max_step = len(true_y)
#     print("plot coreset nodes......")
#     for i in tqdm(range(n_nodes)):
#         filename = os.path.join(path, 'pred_{}point.png'.format(i))
#         time_step = np.arange(0, max_step, 1)
#         fig = plt.figure(figsize=(18, 6))
#         ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
#         if i in conflict_list:
#             ax.plot(time_step, true_y[:max_step, 0, i], color="red", linewidth=1)
#         elif i in stable_list:
#             ax.plot(time_step, true_y[:max_step, 0, i], color="green", linewidth=1)
#         elif i in replay_list:
#             ax.plot(time_step, true_y[:max_step, 0, i], color="magenta", linewidth=1)
#         else:
#             ax.plot(time_step, true_y[:max_step, 0, i], color="black", linewidth=1)
#         plt.savefig(filename, bbox_inches='tight')
#         # plt.show()