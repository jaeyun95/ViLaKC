import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, noutput, dropout):
        super(GCN, self).__init__()
        # self.gc1 = GraphConvolution(nfeat, noutput, None)
        self.gc1 = GraphConvolution(nfeat, nhid, None)
        self.gc2 = GraphConvolution(nhid, noutput, None)
        self.gc3 = GraphConvolution(noutput, noutput, None)

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        return x
