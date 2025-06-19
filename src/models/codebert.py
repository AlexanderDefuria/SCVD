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
