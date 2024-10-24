import time

import torch
import numpy as np
import pandas as pd
import lightning as L

from .datasets import UKBSnpLevelDatasetH5
from .vnn_trainer import GenoVNNLightning, FastVNNLightning
from .graphs import GeneOntology

from dataclasses import make_dataclass


def timing_batches(model, gpu=False, n_repeats=5, batch_sizes=[8, 64, 256, 512]):
    for batch_size in batch_sizes:  # 1024, 2048, 4096]:
        print(f"Trying batch size: {batch_size}")

        total_time = 0.0

        for _ in range(n_repeats):
            # Allocate tensor and move it to the GPU
            x = torch.randn(batch_size, 429371, 1)
            if gpu:
                x = x.to("cuda")

            # Start the timer
            start_time = time.time()

            # Run your model
            model(x)

            # Stop the timer
            end_time = time.time()

            # Calculate the elapsed time (in seconds) and accumulate
            elapsed_time = end_time - start_time
            total_time += elapsed_time

        # Compute average time per batch size
        avg_time_per_batch = total_time / n_repeats

        # Compute time per sample
        time_per_sample = avg_time_per_batch / batch_size

        print(
            f"Average time for batch size {batch_size}: {avg_time_per_batch:.4f} seconds"
        )
        print(
            f"Time per sample for batch size {batch_size}: {time_per_sample:.6f} seconds\n"
        )


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
        "activation": "leaky_relu",
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

    print("Timing on CPU")
    timing_batches(go_vnn_model, gpu=False, batch_sizes=[8, 64, 256, 512, 1024, 4096])

    print("Timing on GPU")
    timing_batches(go_vnn_model.to("cuda"), gpu=True)


if __name__ == "__main__":
    main()
