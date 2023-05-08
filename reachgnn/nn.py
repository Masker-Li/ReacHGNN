import torch
import os
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_geometric.nn.inits import glorot_orthogonal
from torch.nn.functional import silu as swish
import os.path as osp
import urllib.request
from torch_geometric.nn import GATConv, GATv2Conv, HeteroConv, NNConv, GraphNorm
from torch_geometric.nn import radius_graph
from torch_geometric.nn.aggr import Set2Set
from torch_geometric.loader import DataLoader
from torch_geometric.data.makedirs import makedirs
from torch_geometric.utils import remove_self_loops
import warnings
from itertools import product
import numpy as np
from torch import nn
from collections import defaultdict
from torch_scatter import scatter


class BNLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim, act=swish):
        super().__init__()
        self.act = act
        self.lin = Linear(in_dim, out_dim)
        self.bn = torch.nn.BatchNorm1d(out_dim)
        
        torch.nn.init.xavier_uniform_(self.lin.weight)
        #self.lin.bias.data.fill_(0)
        
    def forward(self, x):
        x = self.act(self.bn(self.lin(x)))
        return x

    
class Output_blook(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, scale=1, act=swish):
        super().__init__()
        self.scale = scale
        self.act = act
        self.lin1 = Linear(in_dim, hidden_dim*4)
        self.lin2 = Linear(hidden_dim*4, hidden_dim*2)
        self.lin3 = Linear(hidden_dim*2, out_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim*4)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim*2) 
        
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        #self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        #self.lin2.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin3.weight)
        #self.lin3.bias.data.fill_(0)
        
    def forward(self, x):
        x = self.act(self.bn1(self.lin1(x)))
        x = self.act(self.bn2(self.lin2(x)))
        x = self.scale*self.lin3(x)
        return x


class ReacHGNN(torch.nn.Module):
    def __init__(self, hetero_metadata, x_in_dim, edge_in_dim, num_pre_mpnn=2, num_blocks=2, 
                 hidden_dim=128, out_hidden_dim=512, act=swish, readout: str = 'max'):
        super().__init__()
        self.hetero_metadata = hetero_metadata
        self.x_in_dim = x_in_dim
        self.edge_in_dim = edge_in_dim
        self.num_pre_mpnn = num_pre_mpnn
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.out_hidden_dim = out_hidden_dim
        self.act = act
        self.readout = readout

        # remove dipole_moment

        #self.Reactants = [x for x in self.hetero_metadata[0] if x not in ['Product', 'rxn']]
        self.Reactants = [x for x in self.hetero_metadata[0] if x not in ['rxn']]
        self.num_Reactants = len(self.Reactants)
        
        self.lin0 = Linear(self.x_in_dim, hidden_dim)
        edge_network = Sequential(
            Linear(self.edge_in_dim, hidden_dim*2), nn.ReLU(),
            Linear(hidden_dim*2, hidden_dim * hidden_dim))
        self.conv = NNConv(hidden_dim, hidden_dim, edge_network, aggr='mean', root_weight=True)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.gn = GraphNorm(hidden_dim)
        self.set2set = Set2Set(hidden_dim, processing_steps=3)
        
        inter_layer_dict = {edge: GATv2Conv(hidden_dim, hidden_dim//2, 2, dropout=0.1, add_self_loops=False) 
                              for edge in self.hetero_metadata[1] if (edge[0] != edge[-1] and 
                                                    edge[0] in self.Reactants and edge[-1] in self.Reactants)}
        self.inter_conv = HeteroConv(inter_layer_dict, aggr='sum')

        #self.bnlinear1 = BNLinear(hidden_dim*2, hidden_dim*2, self.act)
        self.bnlinear1 = BNLinear(hidden_dim*4, out_hidden_dim, self.act)
        self.bnlinear2 = BNLinear(hidden_dim*self.num_Reactants, out_hidden_dim, self.act)
        self.bnlinear3 = BNLinear(hidden_dim*self.num_Reactants, out_hidden_dim, self.act)
        self.bnlinear4 = BNLinear(hidden_dim*2*self.num_Reactants, out_hidden_dim, self.act)
        #self.outblock = Output_blook(hidden_dim*10, 1, hidden_dim, self.scale, self.act)
        self.outblock = Sequential(
            Linear(out_hidden_dim*4, out_hidden_dim), nn.PReLU(), nn.Dropout(0.1),
            Linear(out_hidden_dim, out_hidden_dim), nn.PReLU(), nn.Dropout(0.1),
            Linear(out_hidden_dim, 2)
        )

    def forward(self, batch):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict
        
        h_dict = {}
        for i, node_type in enumerate(self.Reactants):
            edge_type = (node_type, 'bond', node_type)
            out = F.relu(self.lin0(batch[node_type].x))
            h = out.unsqueeze(0)
            for _ in range(self.num_pre_mpnn):
                m = F.relu(self.conv(out, edge_index_dict[edge_type], edge_attr_dict[edge_type]))
                out, h = self.gru(m.unsqueeze(0), h)
                out = out.squeeze(0)
            x_dict[node_type], h_dict[node_type] = out, h
           
        for j in range(self.num_blocks):
            x_dict_ = self.inter_conv(x_dict, edge_index_dict)
            for key, value in x_dict_.items():
                x_dict[key] = x_dict[key] + value
            for i, node_type in enumerate(self.Reactants):
                edge_type = (node_type, 'bond', node_type)
                out, h = x_dict[node_type], h_dict[node_type]
                m = F.relu(self.conv(out, edge_index_dict[edge_type], edge_attr_dict[edge_type]))
                out, h = self.gru(m.unsqueeze(0), h)
                out = out.squeeze(0)
                x_dict[node_type], h_dict[node_type] = out, h

        outs1, outs2, outs3, outs4 = [], [], [], []
        x_prod, _batch_prod = x_dict['Product'], batch['Product'].batch
        x_prod = self.gn(x_prod, _batch_prod)
        for i, node_type in enumerate(self.Reactants):
            x, _batch = x_dict[node_type], batch[node_type].batch
            x = self.gn(x, _batch)
            if i < 2:
                react_idx = batch['rxn']['reac_site'][:, i]
                outs1.append(x.index_select(0, react_idx))
                prod_idx = batch['rxn']['prod_site'][:, i]
                outs1.append(x_prod.index_select(0, prod_idx))
            outs2.append(scatter(x, _batch, dim=0, reduce='max'))
            outs3.append(scatter(x, _batch, dim=0, reduce='mean'))
            outs4.append(self.set2set(x, _batch))

        out1 = self.bnlinear1(torch.cat(outs1, dim=1))
        out2 = self.bnlinear2(torch.cat(outs2, dim=1))
        out3 = self.bnlinear3(torch.cat(outs3, dim=1))
        out4 = self.bnlinear4(torch.cat(outs4, dim=1))
        out = self.outblock(torch.cat((out1, out2, out3, out4), dim=1))
        return out[:,0].unsqueeze(0).T, out[:,1].unsqueeze(0).T


class ReacHGNN_noProd(torch.nn.Module):
    def __init__(self, hetero_metadata, x_in_dim, edge_in_dim, num_pre_mpnn=2, num_blocks=2, 
                 hidden_dim=128, out_hidden_dim=512, act=swish, readout: str = 'max'):
        super().__init__()
        self.hetero_metadata = hetero_metadata
        self.x_in_dim = x_in_dim
        self.edge_in_dim = edge_in_dim
        self.num_pre_mpnn = num_pre_mpnn
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.out_hidden_dim = out_hidden_dim
        self.act = act
        self.readout = readout

        # remove dipole_moment

        #self.Reactants = [x for x in self.hetero_metadata[0] if x not in ['Product', 'rxn']]
        self.Reactants = [x for x in self.hetero_metadata[0] if x not in ['rxn']]
        self.num_Reactants = len(self.Reactants)
        
        self.lin0 = Linear(self.x_in_dim, hidden_dim)
        edge_network = Sequential(
            Linear(self.edge_in_dim, hidden_dim*2), nn.ReLU(),
            Linear(hidden_dim*2, hidden_dim * hidden_dim))
        self.conv = NNConv(hidden_dim, hidden_dim, edge_network, aggr='mean', root_weight=True)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.gn = GraphNorm(hidden_dim)
        self.set2set = Set2Set(hidden_dim, processing_steps=3)
        
        inter_layer_dict = {edge: GATv2Conv(hidden_dim, hidden_dim//2, 2, dropout=0.1, add_self_loops=False) 
                              for edge in self.hetero_metadata[1] if (edge[0] != edge[-1] and 
                                                    edge[0] in self.Reactants and edge[-1] in self.Reactants)}
        self.inter_conv = HeteroConv(inter_layer_dict, aggr='sum')

        #self.bnlinear1 = BNLinear(hidden_dim*2, hidden_dim*2, self.act)
        #self.bnlinear1 = BNLinear(hidden_dim*4, hidden_dim*8, self.act)
        self.bnlinear2 = BNLinear(hidden_dim*self.num_Reactants, out_hidden_dim, self.act)
        self.bnlinear3 = BNLinear(hidden_dim*self.num_Reactants, out_hidden_dim, self.act)
        self.bnlinear4 = BNLinear(hidden_dim*2*self.num_Reactants, out_hidden_dim, self.act)
        #self.outblock = Output_blook(hidden_dim*10, 1, hidden_dim, self.scale, self.act)
        self.outblock = Sequential(
            Linear(out_hidden_dim*3, out_hidden_dim), nn.PReLU(), nn.Dropout(0.1),
            Linear(out_hidden_dim, out_hidden_dim), nn.PReLU(), nn.Dropout(0.1),
            Linear(out_hidden_dim, 2)
        )

    def forward(self, batch):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict
        
        h_dict = {}
        for i, node_type in enumerate(self.Reactants):
            edge_type = (node_type, 'bond', node_type)
            out = F.relu(self.lin0(batch[node_type].x))
            h = out.unsqueeze(0)
            for _ in range(self.num_pre_mpnn):
                m = F.relu(self.conv(out, edge_index_dict[edge_type], edge_attr_dict[edge_type]))
                out, h = self.gru(m.unsqueeze(0), h)
                out = out.squeeze(0)
            x_dict[node_type], h_dict[node_type] = out, h
           
        for j in range(self.num_blocks):
            x_dict_ = self.inter_conv(x_dict, edge_index_dict)
            for key, value in x_dict_.items():
                x_dict[key] = x_dict[key] + value
            for i, node_type in enumerate(self.Reactants):
                edge_type = (node_type, 'bond', node_type)
                out, h = x_dict[node_type], h_dict[node_type]
                m = F.relu(self.conv(out, edge_index_dict[edge_type], edge_attr_dict[edge_type]))
                out, h = self.gru(m.unsqueeze(0), h)
                out = out.squeeze(0)
                x_dict[node_type], h_dict[node_type] = out, h

        outs1, outs2, outs3, outs4 = [], [], [], []
        #x_prod, _batch_prod = x_dict['Product'], batch['Product'].batch
        #x_prod = self.gn(x_prod, _batch_prod)
        for i, node_type in enumerate(self.Reactants):
            x, _batch = x_dict[node_type], batch[node_type].batch
            x = self.gn(x, _batch)
            #if i < 2:
                #react_idx = batch['rxn']['reac_site'][:, i]
                #outs1.append(x.index_select(0, react_idx))
                #prod_idx = batch['rxn']['prod_site'][:, i]
                #outs1.append(x_prod.index_select(0, prod_idx))
            outs2.append(scatter(x, _batch, dim=0, reduce='max'))
            outs3.append(scatter(x, _batch, dim=0, reduce='mean'))
            outs4.append(self.set2set(x, _batch))

        #out1 = self.bnlinear1(torch.cat(outs1, dim=1))
        out2 = self.bnlinear2(torch.cat(outs2, dim=1))
        out3 = self.bnlinear3(torch.cat(outs3, dim=1))
        out4 = self.bnlinear4(torch.cat(outs4, dim=1))
        out = self.outblock(torch.cat((out2, out3, out4), dim=1))
        return out[:,0].unsqueeze(0).T, out[:,1].unsqueeze(0).T
