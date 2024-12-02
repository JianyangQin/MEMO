import numpy as np 
import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.gcn_conv import BatchGCNConv


class Basic_Model(nn.Module):
    """Some Information about Basic_Model"""
    def __init__(self, args):
        super(Basic_Model, self).__init__()
        self.dropout = args.dropout
        self.gcn1 = BatchGCNConv(args.gcn["in_channel"], args.gcn["hidden_channel"], bias=True, gcn=False)
        self.gcn2 = BatchGCNConv(args.gcn["hidden_channel"], args.gcn["out_channel"], bias=True, gcn=False)
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.gcn_activation = None
        if args.gcn["activation"]:
            self.gcn_activation = nn.ReLU()
        if args.gcn["fc"]:
            self.activation = nn.GELU()
            self.mlp = nn.Linear(args.gcn['out_channel'], args.y_len)

        self.args = args

    def forward(self, inp, adj):
        N = adj.shape[0]

        if self.gcn_activation:
            x = self.gcn_activation(self.gcn1(inp, adj))           # [bs, N, feature]
        else:
            x = self.gcn1(inp, adj)
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]

        x = self.tcn1(x)                                           # [bs * N, 1, feature]

        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj)                                      # [bs, N, feature]

        x = x + inp
        if self.args.gcn["fc"]:
            x = self.mlp(self.activation(x))
        return x


class Basic_GCN(nn.Module):
    def __init__(self, args):
        super(Basic_GCN, self).__init__()
        self.args = args
        self.dropout = args.dropout

        # spatial temporal encoder
        self.encoder = Basic_Model(args)

        # traffic flow prediction branch
        if args.activation == 'relu':
            self.activation = nn.ReLU()
        elif args.activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = None
        self.mlp = nn.Linear(args.gcn['out_channel'], args.y_len)

    def forward(self, inp, adj, mapping):
        N_adj = adj.shape[0]
        N_pred = len(mapping)

        out = self.encoder(inp, adj)  # view1: n,l,v,c; graph: v,

        # if self.args.pre_scalar:
        #     out = self.activation(self.mlp(out))
        # else:
        out = self.mlp(self.activation(out))[:, mapping, :]
        # out = F.dropout(out, p=self.dropout, training=self.training)

        out = out.reshape(-1, N_pred * self.args.y_len)

        return out

    def predict(self, inp, adj, mapping):
        N = adj.shape[0]
        out = self.forward(inp, adj, mapping)
        out = out.reshape(-1, len(mapping), self.args.y_len)
        out = out.reshape(-1, self.args.y_len)
        return out

    def feature(self, inp, adj):

        z = self.encoder(inp, adj)

        return z


class PECPM(nn.Module):
    def __init__(self, args):
        super(PECPM, self).__init__()
        self.args = args
        self.dropout = args.dropout

        # spatial temporal encoder
        self.encoder = Basic_Model(args)

        self.d = torch.sqrt(torch.Tensor([self.args.gcn['out_channel']]))
        self.w = nn.Parameter(torch.FloatTensor(args.n_clusters, args.gcn['out_channel']), requires_grad=True)  # representation weights
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

        # self.w1 = nn.Linear(args.gcn['out_channel'], args.n_clusters)
        # self.w2 = nn.Linear(args.n_clusters, args.gcn['out_channel'])

        # traffic flow prediction branch
        if args.activation == 'relu':
            self.activation = nn.ReLU()
        elif args.activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = None
        self.mlp = nn.Linear(args.gcn['out_channel'], args.y_len)

        self.loglilike = None

    def forward(self, inp, adj, mapping):
        N_adj = adj.shape[0]
        N_pred = len(mapping)

        z = self.encoder(inp, adj)  # view1: n,l,v,c; graph: v,

        loglilike = torch.softmax(torch.matmul(z, self.w.T) / self.d.to(inp.device), dim=-1)
        self.loglilike = loglilike[:, mapping, :]
        h = torch.matmul(loglilike, self.w)

        # loglilike = torch.softmax(self.w1(z), dim=-1)
        # h = self.w2(loglilike)

        out = h + z

        # if self.args.pre_scalar:
        #     out = self.activation(self.mlp(out))
        # else:
        out = self.mlp(self.activation(out))[:, mapping, :]
        # out = F.dropout(out, p=self.dropout, training=self.training)

        out = out.reshape(-1, N_pred * self.args.y_len)

        return out

    def predict(self, inp, adj, mapping):
        N = adj.shape[0]
        out = self.forward(inp, adj, mapping)
        out = out.reshape(-1, len(mapping), self.args.y_len)
        out = out.reshape(-1, self.args.y_len)
        return out, self.loglilike

    def feature(self, inp, adj):

        z = self.encoder(inp, adj)

        loglilike = torch.softmax(torch.matmul(z, self.w.T) / self.d.to(inp.device), dim=-1)
        h = torch.matmul(loglilike, self.w)

        # loglilike = torch.softmax(self.w1(z), dim=-1)
        # h = self.w2(loglilike)

        z = h + z

        return z


from transformers.models.gpt2.modeling_gpt2 import GPT2Model


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, gcn=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_neigh = nn.Linear(in_features, out_features, bias=bias)
        if not gcn:
            self.weight_self = nn.Linear(in_features, out_features, bias=False)
        else:
            self.register_parameter('weight_self', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_neigh.reset_parameters()
        if self.weight_self is not None:
            self.weight_self.reset_parameters()

    def forward(self, x, adj):
        # x: [bs, N, in_features], adj: [N, N]
        input_x = torch.matmul(adj, x)  # [N, N] * [bs, N, in_features] = [bs, N, in_features]
        output = self.weight_neigh(
            input_x.permute(0, 2, 3, 1))  # [bs, N, in_features] * [in_features, out_features] = [bs, N, out_features]
        return output.permute(0, 3, 1, 2)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, output, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, output)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # print(x.shape)
        x = self.gc2(x, adj)
        # print(x.shape)
        # print('a', F.log_softmax(x, dim=1).shape)
        return F.log_softmax(x, dim=1)

class GPT4TS(nn.Module):
    def __init__(self, gpt_layers=6):
        super(GPT4TS, self).__init__()
        # self.gpt2 = GPT2Model.from_pretrained(
        #     "gpt2", output_attentions=True, output_hidden_states=True
        # )  # loads a pretrained GPT-2 base model
        self.gpt2 = GPT2Model.from_pretrained(
            "/root/JianyangQin/GPT2", output_attentions=True, output_hidden_states=True
        )  # loads a pretrained GPT-2 base model
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        # print("gpt2 = {}".format(self.gpt2))

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if "ln" in name or "wpe" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        return self.gpt2(inputs_embeds=x).last_hidden_state

class STLLM(nn.Module):
    def __init__(
        self,
        args
        # device,
        # adj,
        # input_dim=3,
        # channels=64,
        # num_nodes=170,
        # input_len=12,
        # output_len=12,
        # dropout=0.1,
    ):
        super().__init__()

        # attributes
        self.node_dim = args.gcn["hidden_channel"]
        self.input_len = args.x_len
        self.input_dim = 1
        self.output_len = args.y_len

        gpt_channel = 768

        self.start_conv = nn.Conv2d(
            self.input_dim * self.input_len, gpt_channel, kernel_size=(1, 1)
        )

        self.gcn = GCN(gpt_channel, self.node_dim, gpt_channel, dropout=args.dropout)

        self.gpt = GPT4TS(gpt_layers=3)

        self.regression_layer = nn.Conv2d(
            gpt_channel, self.output_len, kernel_size=(1, 1)
        )

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data, adj, mapping):
        N_adj = adj.shape[-1]
        N_pred = len(mapping)


        # history_data = history_data.x
        history_data = history_data.reshape(-1, N_adj, self.input_len, 1)
        history_data = history_data.permute(0, 2, 1, 3).contiguous()

        input_data = history_data
        batch_size, _, num_nodes, _ = input_data.shape

        input_data = input_data.transpose(1, 2).contiguous()
        input_data = (
            input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        )

        input_data = self.start_conv(input_data)

        data_st = self.gcn(input_data, adj)[:, :, mapping, :] + input_data[:, :, mapping, :]

        data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)
        data_st = self.gpt(data_st)
        data_st = data_st.permute(0, 2, 1).unsqueeze(-1)

        prediction = self.regression_layer(data_st).squeeze(-1)  # [64, 12, 170, 1]
        prediction = prediction.permute(0, 2, 1).reshape(-1, N_pred * self.output_len)

        return prediction


    def predict(self, inp, adj, mapping):
        out = self.forward(inp, adj, mapping)
        out = out.reshape(-1, len(mapping), self.output_len)
        out = out.reshape(-1, self.output_len)
        return out
