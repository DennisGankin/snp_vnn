import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import util
from training_data_wrapper import *
from drugcell_nn import *
from ccc_loss import *

import lightning as L
from torchmetrics.classification import BinaryAccuracy
from torchmetrics import AUROC, ConfusionMatrix

class VNNLightningModel(L.LightningModule):
	def __init__(self, data_wrapper):
		super().__init__()
		self.data_wrapper = data_wrapper
		self.model = DrugCellNN(self.data_wrapper)

		self.loss = CCCLoss()
		# loss as BCE loss
		self.loss = nn.BCELoss()
		# accuracy metric
		self.acc = BinaryAccuracy()

		term_mask_map = util.create_term_mask(self.model.term_direct_gene_map, self.model.gene_dim, self.data_wrapper.cuda)
		for name, param in self.model.named_parameters():
			term_name = name.split('_')[0]
			if '_direct_gene_layer.weight' in name:
				param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
			else:
				param.data = param.data * 0.1
	
	def configure_optimizers(self):
		optimizer = optim.AdamW(self.model.parameters(), lr=self.data_wrapper.lr, betas=(0.9, 0.99), eps=1e-05, weight_decay=self.data_wrapper.lr)
		return optimizer

	def forward(self, x):
		return self.model(x)

	def training_step(self, batch, batch_idx):
		inputs, targets = batch
		#features = util.build_input_vector(inputdata, self.data_wrapper.input_features) # TODO: is this even needed?
		aux_out_map, _ = self.model(inputs)
		output = aux_out_map['final'].squeeze(1)
		loss = self.loss(output, targets)
		# log loss and accuracy
		self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		# calculate accuracy
		acc = self.acc(output, targets)
		self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return loss

	def validation_step(self, batch, batch_idx):
		inputs, targets = batch
		aux_out_map, _ = self.model(inputs)
		output = aux_out_map['final'].squeeze(1)
		loss = self.loss(output, targets)
		# log loss and accuracy
		self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		# calculate accuracy
		acc = self.acc(output, targets)
		self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return loss

class GenoVNNLightning(L.LightningModule):
	def __init__(self, data_wrapper):
		super().__init__()
		self.data_wrapper = data_wrapper
		self.model = GenoVNN(self.data_wrapper)
		# save hyper prams 
		#self.save_hyperparameters()

		# compute class weights
		self.num_positive_samples = torch.sum(self.data_wrapper.dataset.labels == 1).float()
		self.num_negative_samples = torch.sum(self.data_wrapper.dataset.labels == 0).float()
		#self.total_samples = len(self.data_wrapper.dataset.labels)
		#assert self.total_samples == self.num_positive_samples + self.num_negative_samples
		# maybe add class weights to dataset?

		#self.pos_weight = self.total_samples / self.num_positive_samples
		#self.neg_weight = self.total_samples / self.num_negative_samples
		self.pos_weight = self.num_negative_samples / self.num_positive_samples

		#self.weights = torch.tensor([self.neg_weight, self.pos_weight])
        
        # Calculate weighted BCE loss
		#self.loss = F.binary_cross_entropy_with_logits

		#self.loss = CCCLoss()
		# loss as BCE loss
		self.loss = nn.BCELoss()
		#self.loss = nn.BCEWithLogitsLoss(pos_weight=self.weights)
		# accuracy metric
		self.acc = BinaryAccuracy()
		# AUROC metric 
		self.auroc = AUROC(task="binary")
		self.conf_matrix = ConfusionMatrix(task="binary", num_classes=2)

		#self.save_hyperparameters("loss", "optimizer")
	
	def configure_optimizers(self):
		optimizer = optim.AdamW(self.model.parameters(), lr=self.data_wrapper.lr)#, betas=(0.9, 0.99), eps=1e-05, weight_decay=self.data_wrapper.lr)
		scheduler = StepLR(optimizer, step_size=self.data_wrapper.lr_step_size, gamma=0.1)
		return {
			"optimizer": optimizer,
			"lr_scheduler": scheduler,
		}

	def forward(self, x):
		return self.model(x)

	def training_step(self, batch, batch_idx):
		inputs, targets = batch
		#features = util.build_input_vector(inputdata, self.data_wrapper.input_features) # TODO: is this even needed?
		aux_out_map, _ = self.model(inputs)
		output = aux_out_map['final'].squeeze(1)
		#output_logits = aux_out_map['final_logits'].squeeze(1)
		#loss = self.loss(output_logits, targets, pos_weight=self.pos_weight)
		loss = self.loss(output, targets)
		# log loss and accuracy
		self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		# calculate accuracy
		preds = (output > 0.5).float()
		acc = self.acc(preds, targets)
		self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return loss

	def validation_step(self, batch, batch_idx):
		inputs, targets = batch
		aux_out_map, _ = self.model(inputs)
		output = aux_out_map['final'].squeeze(1)
		#output_logits = aux_out_map['final_logits'].squeeze(1)
		#loss = self.loss(output_logits, targets, pos_weight=self.pos_weight)
		loss = self.loss(output, targets)
		# log loss and accuracy
		self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		# calculate accuracy
		preds = (output > 0.5).float()
		acc = self.acc(preds, targets)
		self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
	    # Compute AUC
		auc = self.auroc(output, targets)
		self.log('val_auc', auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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


class VNNTrainer():

	def __init__(self, data_wrapper):
		self.data_wrapper = data_wrapper
		self.train_feature = self.data_wrapper.train_feature
		self.train_label = self.data_wrapper.train_label
		self.val_feature = self.data_wrapper.val_feature
		self.val_label = self.data_wrapper.val_label


	def train_model(self):

		self.model = DrugCellNN(self.data_wrapper)
		self.model.cuda(self.data_wrapper.cuda)

		epoch_start_time = time.time()
		min_loss = None

		term_mask_map = util.create_term_mask(self.model.term_direct_gene_map, self.model.gene_dim, self.data_wrapper.cuda)
		for name, param in self.model.named_parameters():
			term_name = name.split('_')[0]
			if '_direct_gene_layer.weight' in name:
				param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
			else:
				param.data = param.data * 0.1

		train_loader = du.DataLoader(du.TensorDataset(self.train_feature, self.train_label), batch_size=self.data_wrapper.batchsize, shuffle=True, drop_last=True)
		val_loader = du.DataLoader(du.TensorDataset(self.val_feature, self.val_label), batch_size=self.data_wrapper.batchsize, shuffle=True)

		optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.data_wrapper.lr, betas=(0.9, 0.99), eps=1e-05, weight_decay=self.data_wrapper.lr)
		optimizer.zero_grad()

		print("epoch\ttrain_corr\ttrain_loss\ttrue_auc\tpred_auc\tval_corr\tval_loss\tgrad_norm\telapsed_time")
		for epoch in range(self.data_wrapper.epochs):
			# Train
			self.model.train()
			train_predict = torch.zeros(0, 0).cuda(self.data_wrapper.cuda)
			_gradnorms = torch.empty(len(train_loader)).cuda(self.data_wrapper.cuda) # tensor for accumulating grad norms from each batch in this epoch

			for i, (inputdata, labels) in enumerate(train_loader):
				# Convert torch tensor to Variable
				features = util.build_input_vector(inputdata, self.data_wrapper.input_features)
				cuda_features = Variable(features.cuda(self.data_wrapper.cuda))
				cuda_labels = Variable(labels.cuda(self.data_wrapper.cuda))

				# Forward + Backward + Optimize
				optimizer.zero_grad()  # zero the gradient buffer

				aux_out_map,_ = self.model(cuda_features)

				if train_predict.size()[0] == 0:
					train_predict = aux_out_map['final'].data
					train_label_gpu = cuda_labels
				else:
					train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)
					train_label_gpu = torch.cat([train_label_gpu, cuda_labels], dim=0)

				total_loss = 0
				for name, output in aux_out_map.items():
					loss = CCCLoss()
					if name == 'final':
						total_loss += loss(output, cuda_labels)
					else:
						total_loss += self.data_wrapper.alpha * loss(output, cuda_labels)
				total_loss.backward()

				for name, param in self.model.named_parameters():
					if '_direct_gene_layer.weight' not in name:
						continue
					term_name = name.split('_')[0]
					param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

				_gradnorms[i] = util.get_grad_norm(self.model.parameters(), 2.0).unsqueeze(0) # Save gradnorm for batch
				optimizer.step()

			gradnorms = sum(_gradnorms).unsqueeze(0).cpu().numpy()[0] # Save total gradnorm for epoch
			train_corr = util.pearson_corr(train_predict, train_label_gpu)

			self.model.eval()

			val_predict = torch.zeros(0, 0).cuda(self.data_wrapper.cuda)

			val_loss = 0
			for i, (inputdata, labels) in enumerate(val_loader):
				# Convert torch tensor to Variable
				features = util.build_input_vector(inputdata, self.data_wrapper.input_features)
				cuda_features = Variable(features.cuda(self.data_wrapper.cuda))
				cuda_labels = Variable(labels.cuda(self.data_wrapper.cuda))

				aux_out_map, _ = self.model(cuda_features)

				if val_predict.size()[0] == 0:
					val_predict = aux_out_map['final'].data
					val_label_gpu = cuda_labels
				else:
					val_predict = torch.cat([val_predict, aux_out_map['final'].data], dim=0)
					val_label_gpu = torch.cat([val_label_gpu, cuda_labels], dim=0)

				for name, output in aux_out_map.items():
					loss = CCCLoss()
					if name == 'final':
						val_loss += loss(output, cuda_labels)

			val_corr = util.pearson_corr(val_predict, val_label_gpu)

			epoch_end_time = time.time()
			true_auc = torch.mean(train_label_gpu)
			pred_auc = torch.mean(train_predict)
			print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(epoch, train_corr, total_loss, true_auc, pred_auc, val_corr, val_loss, gradnorms, epoch_end_time - epoch_start_time))
			epoch_start_time = epoch_end_time

			if min_loss == None:
				min_loss = val_loss
				torch.save(self.model, self.data_wrapper.modeldir + '/model_final.pt')
				print("Model saved at epoch {}".format(epoch))
			elif min_loss - val_loss > self.data_wrapper.delta:
				min_loss = val_loss
				torch.save(self.model, self.data_wrapper.modeldir + '/model_final.pt')
				print("Model saved at epoch {}".format(epoch))

		return min_loss
