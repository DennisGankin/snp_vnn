import time
import yaml
import argparse

import torch
import numpy as np
import pandas as pd
import lightning as L

from .datasets import UKBSnpLevelDatasetH5, UKBSnpLevelDatasetH5OneHot
from .vnn_trainer import GenoVNNLightning, FastVNNLightning, FastVNNLitReg
from .graphs import GeneOntology

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

import wandb

wandb.login(key="228d1864fd44981291da247f198c331e5cde5ed4")

from dataclasses import make_dataclass


def load_config(yaml_path):
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():

    # Load configuration from YAML file
    parser = argparse.ArgumentParser(
        description="Run model training with custom config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Path to the YAML config file.",
    )
    cmd_args = parser.parse_args()
    config = load_config(cmd_args.config)

    # Dynamically create dataclass from YAML config
    args = make_dataclass(
        "DataclassFromConfig", [(k, type(v)) for k, v in config.items()]
    )(**config)

    ###### load dataset
    if args.onehot:
        print("Using one hot encoding")
        dataset = UKBSnpLevelDatasetH5OneHot(args)
    else:
        print("Using raw data")
        dataset = UKBSnpLevelDatasetH5(args)
    args.feature_dim = dataset.feature_dim  ### TODO

    ##### crate gene ontology object
    graph = GeneOntology(dataset.gene_id_mapping, args.onto, child_node="snp")

    ##### load DL model
    if args.task == "classification":
        print("Classification task")
        go_vnn_model = FastVNNLightning(args, graph)  # GenoVNNLightning(args, graph)
    else:
        print("Regression task")
        go_vnn_model = FastVNNLitReg(args, graph)
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
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    # create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=int(args.num_workers),  # 32,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=int(args.num_workers / 2),
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
        project="vnn_ukb_breast_cancer", name="nest_height" + curr_time
    )
    logger = TensorBoardLogger(
        name="nest_ukb_model_" + curr_time, save_dir="/home/dnanexus/lightning_logs"
    )
    # trainer = L.Trainer(max_epochs=args.epoch, logger=logger)
    trainer = L.Trainer(  # profiler="simple",
        max_epochs=args.epoch,  # max_steps=4,  #
        logger=[logger, wandb_logger],  # val_check_interval=0.25,
        log_every_n_steps=10,
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
