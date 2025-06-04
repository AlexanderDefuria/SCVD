from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Dropout, Sequential, BatchNorm1d
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv, GraphConv, global_mean_pool, Sequential as GeoSequential
from torch_geometric.data import DataLoader
import torchmetrics
from transformers import AutoModelForSequenceClassification

from src.utils import interact


class GNNModel(torch.nn.Module):
    """Graph Neural Network model using PyTorch Geometric."""

    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super(GNNModel, self).__init__()

        # Graph convolution layers
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)

        # Batch normalization
        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)

        # Classification head
        self.classifier = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Dropout(0.5), Linear(hidden_dim, output_dim))

    def forward(self, x, edge_index, batch):
        # First graph convolution block
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

    def __init__(self, input_dim, hidden_dim=64, output_dim=2, learning_rate=0.001):
        super(GraphModel, self).__init__()

        self.save_hyperparameters()
        self.model = GNNModel(input_dim, hidden_dim, output_dim)
        self.learning_rate = learning_rate

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=output_dim)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=output_dim)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=output_dim)

    def forward(self, data):
        # Extract features from the data batch
        x = self.prepare_features(data)

        # Forward pass through the model
        return self.model(x, data.edge_index, data.batch)

    def prepare_features(self, data):
        """
        Extract node features from the data batch.
        This needs to be customized based on your specific data structure.
        """
        # Note: This is a placeholder - you'll need to adapt this to your data
        # For example, if statements contain numerical features:
        if hasattr(data, "statements") and data.statements is not None:
            return data.statements
        # If your node features are stored in a field called 'x':
        elif hasattr(data, "x") and data.x is not None:
            return data.x
        # Otherwise, create one-hot encodings as default features
        else:
            return torch.ones(data.num_nodes, 1)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        y = batch.label
        loss = F.cross_entropy(y_hat, y)

        # Log metrics
        self.train_accuracy(y_hat.softmax(dim=1), y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        y = batch.label
        loss = F.cross_entropy(y_hat, y)

        # Log metrics
        self.val_accuracy(y_hat.softmax(dim=1), y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        y = batch.label
        loss = F.cross_entropy(y_hat, y)

        # Log metrics
        self.test_accuracy(y_hat.softmax(dim=1), y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
        return [optimizer], [scheduler]


class CodeBERTModel(pl.LightningModule):
    # TODO Change CodeBERT to more SOTA model...
    """
    CodeBERT model for source code based vulnerability detection classification.
    """

    def __init__(self, lr: float = 2e-5, num_labels: int = 2):
        super(CodeBERTModel, self).__init__()
        checkpoint = "microsoft/codebert-base"
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2, output_hidden_states=True).to(device="cuda")

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].to(device="cuda")
        attention_mask = batch["attention_mask"].to(device="cuda")
        labels = batch["label"].to(device="cuda")
        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].to(device="cuda")
        attention_mask = batch["attention_mask"].to(device="cuda")
        labels = batch["label"].to(device="cuda")
        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        input_ids = args[0]["input_ids"].to(device="cuda")
        attention_mask = args[0]["attention_mask"].to(device="cuda")
        labels = args[0]["label"].to(device="cuda")
        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, labels)
        self.log("test_loss", loss, prog_bar=True)
        return loss
