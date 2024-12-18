import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, LambdaLR

from . import util

from .models import *

import lightning as L
from torchmetrics.classification import BinaryAccuracy
from torchmetrics import AUROC, ConfusionMatrix, PearsonCorrCoef

from torchvision.ops import sigmoid_focal_loss


class GenoVNNLightning(L.LightningModule):
    def __init__(self, args, graph):
        super().__init__()
        self.args = args
        self.model = GenoVNN(self.args, graph)
        # save hyper params
        # self.save_hyperparameters()

        # compute class weights
        # self.num_positive_samples = torch.sum(
        #    self.data_wrapper.dataset.labels == 1
        # ).float()
        # self.num_negative_samples = torch.sum(
        #    self.data_wrapper.dataset.labels == 0
        # ).float()
        # self.total_samples = len(self.data_wrapper.dataset.labels)
        # assert self.total_samples == self.num_positive_samples + self.num_negative_samples
        # maybe add class weights to dataset?

        # self.pos_weight = self.total_samples / self.num_positive_samples
        # self.neg_weight = self.total_samples / self.num_negative_samples
        # self.pos_weight = self.num_negative_samples / self.num_positive_samples

        # self.weights = torch.tensor([self.neg_weight, self.pos_weight])

        # Calculate weighted BCE loss
        # self.loss = F.binary_cross_entropy_with_logits

        # self.loss = CCCLoss()
        # loss as BCE loss
        self.loss = nn.BCEWithLogitsLoss()
        # self.loss = nn.BCEWithLogitsLoss(pos_weight=self.weights)
        # accuracy metric
        self.acc = BinaryAccuracy()
        # AUROC metric
        self.auroc = AUROC(task="binary")
        self.conf_matrix = ConfusionMatrix(task="binary", num_classes=2)

        # self.save_hyperparameters("loss", "optimizer")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.args.lr, weight_decay=0.1
        )  # , betas=(0.9, 0.99), eps=1e-05, weight_decay=self.data_wrapper.lr)
        scheduler = StepLR(optimizer, step_size=self.args.lr_step_size, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        # features = util.build_input_vector(inputdata, self.data_wrapper.input_features) # TODO: is this even needed?
        aux_out_map, _ = self.model(inputs)
        output = aux_out_map["final"].squeeze(1)
        output_logits = aux_out_map["final_logits"].squeeze(1)
        # loss = self.loss(output_logits, targets, pos_weight=self.pos_weight)
        loss = self.loss(output_logits, targets)
        # log loss and accuracy
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # calculate accuracy
        preds = (output > 0.5).float()
        acc = self.acc(preds, targets)
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        aux_out_map, _ = self.model(inputs)
        output = aux_out_map["final"].squeeze(1)
        output_logits = aux_out_map["final_logits"].squeeze(1)
        # loss = self.loss(output_logits, targets, pos_weight=self.pos_weight)
        loss = self.loss(output_logits, targets)
        # log loss and accuracy
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # calculate accuracy
        preds = (output > 0.5).float()
        acc = self.acc(preds, targets)
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # Compute AUC
        auc = self.auroc(output, targets)
        self.log(
            "val_auc", auc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        # update confusion matrix
        self.conf_matrix.update(preds, targets)
        return loss

    def on_validation_epoch_end(self):
        # Compute the confusion matrix at the end of the validation epoch
        confmat = self.conf_matrix.compute()
        # Log the confusion matrix to TensorBoard
        util.log_confusion_matrix(self, confmat)
        # Reset confusion matrix metric for the next epoch
        self.conf_matrix.reset()


class FastVNNLightning(L.LightningModule):
    def __init__(self, args, graph):
        super().__init__()
        self.args = args
        self.model = FastVNN(self.args, graph)
        # save hyper params
        # self.save_hyperparameters()

        # loss as BCE loss
        # self.loss = nn.BCEWithLogitsLoss()
        self.loss = lambda x, y: sigmoid_focal_loss(x, y, reduction="mean")
        # self.loss = nn.BCEWithLogitsLoss(pos_weight=self.weights)
        # accuracy metric
        self.acc = BinaryAccuracy()
        # AUROC metric
        self.auroc = AUROC(task="binary")
        self.auroc_train = AUROC(task="binary")
        self.conf_matrix = ConfusionMatrix(task="binary", num_classes=2)

        # self.save_hyperparameters("loss", "optimizer")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.args.lr
        )  # , betas=(0.9, 0.99), eps=1e-05, weight_decay=self.data_wrapper.lr)
        scheduler = StepLR(optimizer, step_size=self.args.lr_step_size, gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        # features = util.build_input_vector(inputdata, self.data_wrapper.input_features) # TODO: is this even needed?
        aux_out_map, _ = self.model(inputs)
        output = aux_out_map["final"].squeeze(1)
        output_logits = aux_out_map["final_logits"].squeeze(1)
        # loss = self.loss(output_logits, targets, pos_weight=self.pos_weight)
        loss = self.loss(output_logits, targets)
        # log loss and accuracy
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # calculate AUC
        auc = self.auroc_train(output, targets)
        self.log(
            "train_auc", auc, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
        # calculate accuracy
        preds = (output > 0.5).float()
        acc = self.acc(preds, targets)
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        if batch_idx == 0:
            util.log_boxplots(self, output, output_logits, targets, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        aux_out_map, _ = self.model(inputs)
        output = aux_out_map["final"].squeeze(1)
        output_logits = aux_out_map["final_logits"].squeeze(1)
        # loss = self.loss(output_logits, targets, pos_weight=self.pos_weight)
        loss = self.loss(output_logits, targets)
        # log loss and accuracy
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # calculate accuracy
        preds = (output > 0.5).float()
        acc = self.acc(preds, targets)
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # Compute AUC
        auc = self.auroc(output, targets)
        self.log(
            "val_auc", auc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        auc = self.auroc(output_logits, targets)
        self.log(
            "val_auc_logits",
            auc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        if batch_idx == 0:
            util.log_boxplots(self, output, output_logits, targets, "val")
        # update confusion matrix
        self.conf_matrix.update(preds, targets)
        return loss

    def on_validation_epoch_end(self):
        # Compute the confusion matrix at the end of the validation epoch
        confmat = self.conf_matrix.compute()
        # Log the confusion matrix to TensorBoard
        util.log_confusion_matrix(self, confmat)
        # Reset confusion matrix metric for the next epoch
        self.conf_matrix.reset()


class FastVNNLitReg(L.LightningModule):
    def __init__(self, args, graph):
        super().__init__()
        self.args = args
        self.model = FastVNN(self.args, graph)
        # save hyper params
        # self.save_hyperparameters()

        # MSE loss
        self.loss = nn.MSELoss()
        # Pearson metric
        self.pearson_train = PearsonCorrCoef()
        self.pearson_val = PearsonCorrCoef()

        # self.save_hyperparameters("loss", "optimizer")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.args.lr
        )  # , betas=(0.9, 0.99), eps=1e-05, weight_decay=self.data_wrapper.lr)
        scheduler = StepLR(optimizer, step_size=self.args.lr_step_size, gamma=0.1)
        # Define the lambda function for the learning rate schedule
        # If epoch < 3, return 1.0 (lr * 1.0 = 0.01)
        # If epoch >= 3, return 0.1 (lr * 0.1 = 0.001)
        # lambda_fn = lambda epoch: 1.0 if epoch < 6 else 0.02
        # scheduler = LambdaLR(optimizer, lr_lambda=lambda_fn)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        aux_out_map, _ = self.model(inputs)
        output_logits = aux_out_map["final_logits"].squeeze(1)
        loss = self.loss(output_logits, targets)
        # log loss and accuracy
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # calculate AUC
        pearson = self.pearson_train(output_logits, targets)
        self.log(
            "train_pearson",
            pearson,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        if batch_idx == 0:
            util.log_scatter(self, output_logits, targets, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        aux_out_map, _ = self.model(inputs)
        output_logits = aux_out_map["final_logits"].squeeze(1)
        loss = self.loss(output_logits, targets)
        # log loss and accuracy
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # Compute pearson
        pearson = self.pearson_val(output_logits, targets)
        self.log(
            "val_pearson",
            pearson,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if batch_idx == 0:
            util.log_scatter(self, output_logits, targets, "val")
        # update confusion matrix
        # self.conf_matrix.update(preds, targets)
        return loss
