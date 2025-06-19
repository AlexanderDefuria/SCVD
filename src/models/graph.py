from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ReLU, Dropout, Sequential, BatchNorm1d
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv, GraphConv, global_mean_pool, Sequential as GeoSequential
from torch_geometric.data import DataLoader
import torchmetrics
from transformers import AutoModelForSequenceClassification

from src.graphml import Mappings
from src.utils import interact


def map_into_abstractions(databatch, mapping: Mappings):
    """ """
    x = torch.zeros((databatch.num_nodes, 4), dtype=torch.float)

    idx = 0
    for i, variable in enumerate([databatch.api_call, databatch.operator, databatch.data_type, databatch.constant]):
        for value in variable:
            x[idx : idx + len(value), i] = torch.Tensor(list(map(lambda v: mapping.get(v, 0), value)))  # Map each value to its abstraction index

        idx += len(variable)

    return x


class GNNModel(torch.nn.Module):
    """Graph Neural Network model using PyTorch Geometric."""

    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(GNNModel, self).__init__()

        # Graph convolution layers
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)

        # Batch normalization
        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)

        # Classification head
        self.classifier = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Dropout(0.1), Linear(hidden_dim, output_dim))

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Second graph convolution block
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Third graph convolution block
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Global pooling (from node-level to graph-level)
        x = global_mean_pool(x, batch)

        # Classification
        x = self.classifier(x)

        return x


class GraphModel(pl.LightningModule):
    """PyTorch Lightning module for graph classification."""

    def __init__(
        self,
        input_dim,
        mapping,
        hidden_dim=64,
        output_dim=2,
        learning_rate=0.001,
    ):
        super(GraphModel, self).__init__()

        self.save_hyperparameters(ignore=["mapping"])
        self.model = GNNModel(input_dim, hidden_dim, output_dim)
        self.learning_rate = learning_rate
        self.mapping = mapping

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=output_dim)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=output_dim)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=output_dim)

    def forward(self, data):
        x = map_into_abstractions(data, self.mapping)
        x = x.to(self.device)
        return self.model(x, data.edge_index, data.batch)

    def logging(self, label, metric, batch_size):
        self.log(label, metric, prog_bar=True, sync_dist=True, batch_size=batch_size)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        y = batch.label
        loss = F.cross_entropy(y_hat, y)

        # Log metrics
        self.train_accuracy(y_hat.softmax(dim=1), y)
        self.logging("train_loss", loss, batch_size=batch.num_graphs)
        self.logging("train_acc", self.train_accuracy, batch_size=batch.num_graphs)

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        y = batch.label
        loss = F.cross_entropy(y_hat, y)

        # Log metrics
        self.val_accuracy(y_hat.softmax(dim=1), y)
        self.logging("val_loss", loss, batch_size=batch.num_graphs)
        self.logging("val_acc", self.val_accuracy, batch_size=batch.num_graphs)

        return loss

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        y = batch.label
        loss = F.cross_entropy(y_hat, y)

        # Log metrics
        self.test_accuracy(y_hat.softmax(dim=1), y)
        self.logging("test_loss", loss, batch_size=batch.num_graphs)
        self.logging("test_acc", self.test_accuracy, batch_size=batch.num_graphs)

        return loss

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]
