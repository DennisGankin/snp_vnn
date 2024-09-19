import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import util

from sklearn.preprocessing import StandardScaler


# custom pytroch dataset
class GenotypeDataset(Dataset):
    def __init__(self, args):
        # load the data
        # self.sample_id_mapping = util.load_mapping(args.cell2id, 'samples')
        # self.gene_id_mapping = util.load_mapping(args.gene2id, 'genes (input features)')

        # load input features
        self.mutations = np.load(args.mutations)["bin_features"]
        self.data = torch.from_numpy(np.dstack([self.mutations])).float()

        # load labels
        self.label_df = pd.read_csv(
            args.train,
            sep="\t",
            header=None,
            names=["cell_line", "smiles", "auc", "dataset"],
        )
        self.labels = torch.from_numpy(self.label_df["auc"].values).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class UKBGeneLevelDataset(Dataset):
    def __init__(self, args):
        # load labels
        self.label_col = args.label_col
        self.label_df = pd.read_csv(args.train)
        self.labels = torch.from_numpy(self.label_df[self.label_col].values).float()

        # load input features
        # self.perturbed_genes = np.load(args.mutations)["bin_features"]
        self.perturbed_genes = np.load(args.mutations)  # ["bin_features"]
        # Standardize the perturbed_genes
        scaler = StandardScaler()
        self.perturbed_genes = scaler.fit_transform(self.perturbed_genes)
        # self.perturbed_genes = self.perturbed_genes>0
        # load gene names
        self.gene_df = pd.read_csv(args.gene2id).reset_index()
        self.gene_id_mapping = {
            gene: idx
            for idx, gene in zip(self.gene_df.index, self.gene_df["gene_name"])
        }
        # filter perturbed_genes
        self.perturbed_genes = self.perturbed_genes[:, self.gene_df["gene_bed_id"]]
        self.data = torch.from_numpy(np.dstack([self.perturbed_genes])).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
