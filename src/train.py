import time

import torch
import numpy as np
import pandas as pd
import lightning as L

from .datasets import UKBSnpLevelDatasetH5
from .vnn_trainer import GenoVNNLightning, FastVNNLightning
from .graphs import GeneOntology

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

import wandb

wandb.login(key="228d1864fd44981291da247f198c331e5cde5ed4")

from dataclasses import make_dataclass


def main():
    argument_dict = {
        "onto": "../ontology.txt",
        "train": "../labels.csv",
        "label_col": "bc_reported",  # "has_cancer", # new for ukb
        "mutations": "../genotype_data.h5",  # "../merged_allchr.bed",
        "epoch": 5,
        "lr": 0.003,
        "wd": 0.001,
        "alpha": 0.3,
        "batchsize": 512,  # 33840,
        "modeldir": "/model_test/",
        "cuda": 0,
        "gene2id": "../ukb_snp_ids.csv",
        "genotype_hiddens": 4,
        "optimize": 1,
        "zscore_method": "auc",
        "std": "/model_test/std.txt",
        "patience": 30,
        "delta": 0.001,
        "min_dropout_layer": 2,
        "dropout_fraction": 0.3,
        "lr_step_size": 120,
    }
    args = make_dataclass(
        "DataclassFromDir", ((k, type(v)) for k, v in argument_dict.items())
    )(**argument_dict)

    ###### load dataset
    dataset = UKBSnpLevelDatasetH5(args)
    args.feature_dim = dataset.feature_dim  ### TODO

    ##### crate gene ontology object
    graph = GeneOntology(dataset.gene_id_mapping, args.onto, child_node="snp")

    ##### load DL model
    go_vnn_model = FastVNNLightning(args, graph)  # GenoVNNLightning(args, graph)
    pytorch_total_params = sum(
        p.numel() for p in go_vnn_model.parameters() if p.requires_grad
    )
    print("Number of parameters: ", pytorch_total_params)

    # Compile the model
    # go_vnn_model = torch.compile(go_vnn_model)  # triton error

    # for reproducibility, set seed
    torch.manual_seed(700)

    # for speed
    torch.set_float32_matmul_precision("medium")

    # random train validation split
    _, train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [0.95, 0.045, 0.005]
    )
    # create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    print("Data is ready.")

    curr_time = time.strftime("%Y%m%d_%H%M")
    # time how long it takes to train the model

    # log the learning rate
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Initialize WandbLogger
    wandb_logger = WandbLogger(
        project="vnn_ukb_breast_cancer", name="nest_snplevel_" + curr_time
    )
    logger = TensorBoardLogger(
        name="nest_ukb_model_" + curr_time, save_dir="/home/dnanexus/lightning_logs"
    )
    # trainer = L.Trainer(max_epochs=args.epoch, logger=logger)
    trainer = L.Trainer(
        profiler="simple",
        max_steps=4,  # max_epochs=args.epoch,
        logger=[logger, wandb_logger],
        val_check_interval=0.25,
        log_every_n_steps=1,
        precision=16,
        callbacks=[lr_monitor],
    )  # log every steps should depend on the batch size
    # trainer = L.Trainer(max_epochs=args.epoch, logger=[logger, wandb_logger], profiler="simple") # log every steps should depend on the batch size

    print("Starting training")
    # Record the start time
    start_time = time.time()
    # Train the model
    trainer.fit(
        model=go_vnn_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    # Record the end time
    end_time = time.time()
    # Calculate and print the total training time
    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")


if __name__ == "__main__":
    main()
