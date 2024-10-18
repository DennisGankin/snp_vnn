import time

import torch
import numpy as np
import pandas as pd
import lightning as L

from .datasets import UKBSnpLevelDatasetH5
from .vnn_trainer import GenoVNNLightning, FastVNNLightning
from .graphs import GeneOntology

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
        "batchsize": 1480,  # 33840,
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

    print("Setting up data")
    snp_df = pd.read_csv(args.gene2id).reset_index()
    args.feature_dim = 1  # input features per snp

    gene_id_mapping = {  # mapping row in feature matrix to the corresponding snp
        snp: idx for idx, snp in zip(snp_df.index, snp_df["snp"])
    }

    print("Setting up gene ontology")
    ##### crate gene ontology object
    graph = GeneOntology(gene_id_mapping, args.onto, child_node="snp")

    print("Setting up DL model")
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

    # model to device
    go_vnn_model.to("cuda")

    # create a random tensor for testing
    # create tensors of inreasing batch size and test pass through model
    for batch_size in [8, 64, 256, 512]:  # 1024, 2048, 4096]:
        print("Trying batch size: ", batch_size)
        x = torch.randn(batch_size, 429371, 1).to("cuda")
        go_vnn_model(x)


if __name__ == "__main__":
    main()
