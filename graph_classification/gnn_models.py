

# gnn_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool, global_add_pool

# OGB-specific import for feature encoding
try:
    from ogb.graphproppred.mol_encoder import AtomEncoder
except ImportError:
    AtomEncoder = None

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden=64, num_layers=3, dropout=0.5, pool='mean', use_ogb_features=False, use_bn=True, use_residual=True):
        super().__init__()
        self.use_bn = use_bn
        self.use_residual = use_residual
        
        if use_ogb_features and AtomEncoder is not None:
            self.encoder = AtomEncoder(hidden)
        else:
            self.encoder = nn.Linear(num_features, hidden)
        
        input_dim = hidden

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden
            self.convs.append(GCNConv(in_channels, hidden))
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(hidden))
        
        self.lin = torch.nn.Linear(hidden, num_classes)
        self.dropout = dropout
        self.pool = global_mean_pool if pool == 'mean' else global_add_pool
        self.reset_parameters()

    def reset_parameters(self):
        if not isinstance(self.encoder, AtomEncoder):
            self.encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if self.use_bn:
            for bn in self.bns:
                bn.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.encoder(x)
        
        for i, conv in enumerate(self.convs):
            h = x
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual and h.shape == x.shape:
                x = h + x # Residual connection
                
        x = self.pool(x, batch)
        return self.lin(x)


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden=8, heads=8, num_layers=3, dropout=0.6, pool='mean', use_ogb_features=False, use_bn=True, use_residual=True):
        super().__init__()
        self.use_bn = use_bn
        self.use_residual = use_residual
        
        if use_ogb_features and AtomEncoder is not None:
            self.encoder = AtomEncoder(hidden * heads)
            input_dim = hidden * heads
        else:
            self.encoder = nn.Linear(num_features, hidden * heads)
            input_dim = hidden * heads
            
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(GATConv(input_dim, hidden, heads=heads, dropout=dropout))
        if self.use_bn: self.bns.append(nn.BatchNorm1d(hidden * heads))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden * heads, hidden, heads=heads, dropout=dropout))
            if self.use_bn: self.bns.append(nn.BatchNorm1d(hidden * heads))
            
        self.convs.append(GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=dropout))
        if self.use_bn: self.bns.append(nn.BatchNorm1d(hidden))

        self.lin = torch.nn.Linear(hidden, num_classes)
        self.dropout = dropout
        self.pool = global_mean_pool if pool == 'mean' else global_add_pool
        self.reset_parameters()

    def reset_parameters(self):
        if not isinstance(self.encoder, AtomEncoder):
            self.encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if self.use_bn:
            for bn in self.bns:
                bn.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.encoder(x)

        for i, conv in enumerate(self.convs):
            h = x
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            if i < len(self.convs) - 1:
                x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual and h.shape == x.shape:
                x = h + x # Residual connection
                
        x = self.pool(x, batch)
        return self.lin(x)


class SAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden=64, num_layers=3, dropout=0.5, pool='mean', use_ogb_features=False, use_bn=True, use_residual=True):
        super().__init__()
        self.use_bn = use_bn
        self.use_residual = use_residual

        if use_ogb_features and AtomEncoder is not None:
            self.encoder = AtomEncoder(hidden)
        else:
            self.encoder = nn.Linear(num_features, hidden)
        
        input_dim = hidden

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden
            self.convs.append(SAGEConv(in_channels, hidden))
            if self.use_bn:
                self.bns.append(nn.BatchNorm1d(hidden))

        self.lin = torch.nn.Linear(hidden, num_classes)
        self.dropout = dropout
        self.pool = global_mean_pool if pool == 'mean' else global_add_pool
        self.reset_parameters()

    def reset_parameters(self):
        if not isinstance(self.encoder, AtomEncoder):
            self.encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if self.use_bn:
            for bn in self.bns:
                bn.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.encoder(x)
        
        for i, conv in enumerate(self.convs):
            h = x
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual and h.shape == x.shape:
                x = h + x # Residual connection
                
        x = self.pool(x, batch)
        return self.lin(x)

class GIN(torch.nn.Module):
    # GIN already includes batch norm in its standard formulation.
    # Residual connections are less common due to the internal MLP structure.
    def __init__(self, num_features, num_classes, hidden=64, num_layers=3, dropout=0.5, pool='mean', use_ogb_features=False):
        super().__init__()
        if use_ogb_features and AtomEncoder is not None:
            self.encoder = AtomEncoder(hidden)
        else:
            self.encoder = nn.Linear(num_features, hidden)
        
        input_dim = hidden
            
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        mlp_initial = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.convs.append(GINConv(mlp_initial, train_eps=True))
        self.bns.append(nn.BatchNorm1d(hidden))

        for _ in range(num_layers - 1):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden))
            
        self.lin = torch.nn.Linear(hidden, num_classes)
        self.dropout = dropout
        self.pool = global_mean_pool if pool == 'mean' else global_add_pool
        self.reset_parameters()

    def reset_parameters(self):
        if not isinstance(self.encoder, AtomEncoder):
            self.encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.encoder(x)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.pool(x, batch)
        return self.lin(x)