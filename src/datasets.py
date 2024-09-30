import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler
from pysnptools.snpreader import Bed

import h5py


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


class UKBSnpLevelDatasetBED(Dataset):
    def __init__(self, args):
        # load labels
        self.label_col = args.label_col
        self.label_df = pd.read_csv(args.train)
        self.labels = torch.from_numpy(self.label_df[self.label_col].values).float()
        self.bed_ids = self.label_df["bed_id"]

        # load input features from bed file
        self.bed = Bed(args.mutations, count_A1=False)

        # get snp ids in the bed file (TODO: rename)
        self.snp_df = pd.read_csv(args.gene2id).reset_index()
        self.snp_bed_ids = self.snp_df["snp_bed_id"]  # snp id in the bed file
        self.feature_dim = 1  # input features per snp

        self.gene_id_mapping = (
            {  # mapping row in feature matrix to the corresponding snp
                snp: idx for idx, snp in zip(self.snp_df.index, self.snp_df["snp"])
            }
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        subset_bed = self.bed[self.bed_ids[idx], self.snp_bed_ids]
        data = subset_bed.read().val
        data = np.nan_to_num(data).T
        data = torch.from_numpy(data).float()

        return data, self.labels[idx]


class UKBSnpLevelDatasetH5(Dataset):
    """
    Dataset for UKB SNP level data.
    Using HPF5 file format.
    """

    def __init__(self, args):
        # load labels
        self.label_col = args.label_col
        self.label_df = pd.read_csv(args.train)
        self.labels = torch.from_numpy(self.label_df[self.label_col].values).float()

        # load hdf5 file
        self.hdf = h5py.File(args.mutations, "r")

        # get snp ids in the hdf5 file (TODO: rename)
        self.snp_df = pd.read_csv(args.gene2id).reset_index()
        self.snp_bed_ids = self.snp_df["snp_bed_id"].astype(
            str
        )  # snp id in the hdf5 file
        self.feature_dim = 1  # input features per snp

        self.gene_id_mapping = (
            {  # mapping row in feature matrix to the corresponding snp
                snp: idx for idx, snp in zip(self.snp_df.index, self.snp_df["snp"])
            }
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        snp_id = self.snp_bed_ids[idx]
        data = self.hdf[snp_id][:]
        data = np.nan_to_num(data).T
        data = torch.from_numpy(data).float()

        return data, self.labels[idx]
