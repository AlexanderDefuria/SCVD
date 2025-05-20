from pytorch_lightning import LightningModule
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, input_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return 

class GraphModel(LightningModule):
    def __init__(self):
        super().__init__()
        # Define your model here
        self.model = GCN()  # Replace with your model definition
    def forward(self, x):
        # Define the forward pass
        self.model(x)

    def training_step(self, batch, batch_idx):
        # Define the training step
        pass
    def configure_optimizers(self):
        # Define the optimizer
        pass
