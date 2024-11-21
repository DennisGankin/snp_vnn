import torch
from torch import inf
import math
import numpy as np
import pandas as pd
import random as rd
from sklearn.preprocessing import robust_scale
from sklearn.preprocessing import scale
from scipy import stats
from sklearn.metrics import r2_score
import yaml

from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme("talk")

import io

# import wandb


def pearson_corr(x, y):
    xx = x - torch.mean(x)
    yy = y - torch.mean(y)

    return torch.sum(xx * yy) / (torch.norm(xx, 2) * torch.norm(yy, 2))


def spearman_corr(torch_pred, torch_labels):
    pred = torch_pred.cpu().numpy()
    labels = torch_labels.cpu().numpy()
    corr = stats.spearmanr(pred, labels)[0]
    return corr


def get_r2_score(torch_pred, torch_labels):
    pred = torch_pred.cpu().numpy()
    labels = torch_labels.cpu().numpy()
    r2 = r2_score(labels, pred)
    return r2


def get_drug_corr_median(torch_pred, torch_labels, torch_inputdata):

    pred = torch_pred.cpu().numpy()
    labels = torch_labels.cpu().numpy()
    inputdata = torch_inputdata.cpu().numpy()

    drugs = set([int(data[1]) for data in inputdata])

    pos_map = {d: [] for d in drugs}
    for i, data in enumerate(inputdata):
        pos_map[data[1]].append(i)

    corr_list = []
    for drug in drugs:
        index = pos_map[drug]
        x = np.take(pred, index)
        y = np.take(labels, index)
        corr = stats.spearmanr(x, y)[0]
        corr_list.append(corr)

    return np.median(corr_list)


# adapted from https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab
def class_accuracy(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def calc_std_vals(df, zscore_method):
    std_df = pd.DataFrame(columns=["dataset", "center", "scale"])
    std_list = []

    if zscore_method == "zscore":
        for name, group in df.groupby(["dataset"])["auc"]:
            center = group.mean()
            scale = group.std()
            if math.isnan(scale) or scale == 0.0:
                scale = 1.0
            temp = pd.DataFrame([[name, center, scale]], columns=std_df.columns)
            std_list.append(temp)

    elif zscore_method == "robustz":
        for name, group in df.groupby(["dataset"])["auc"]:
            center = group.median()
            scale = group.quantile(0.75) - group.quantile(0.25)
            if math.isnan(scale) or scale == 0.0:
                scale = 1.0
            temp = pd.DataFrame([[name, center, scale]], columns=std_df.columns)
            std_list.append(temp)
    else:
        for name, group in df.groupby(["dataset"])["auc"]:
            temp = pd.DataFrame([[name, 0.0, 1.0]], columns=std_df.columns)
            std_list.append(temp)

    std_df = pd.concat(std_list, ignore_index=True)
    return std_df


def standardize_data(df, std_df):
    merged = pd.merge(df, std_df, how="left", on=["dataset"], sort=False)
    merged["z"] = (merged["auc"] - merged["center"]) / merged["scale"]
    merged = merged[["cell_line", "smiles", "z"]]
    return merged


def load_train_data(train_file, cell2id, zscore_method, std_file):

    all_df = pd.read_csv(
        train_file,
        sep="\t",
        header=None,
        names=["cell_line", "smiles", "auc", "dataset"],
    )

    train_cell_lines = list(set(all_df["cell_line"]))
    val_cell_lines = []
    val_size = int(len(train_cell_lines) / 5)

    for _ in range(val_size):
        r = rd.randint(0, len(train_cell_lines) - 1)
        val_cell_lines.append(train_cell_lines.pop(r))

    val_df = all_df.query("cell_line in @val_cell_lines").reset_index(drop=True)
    train_df = all_df.query("cell_line in @train_cell_lines").reset_index(drop=True)

    std_df = calc_std_vals(train_df, zscore_method)
    std_df.to_csv(std_file, sep="\t", header=False, index=False)
    # train_df = standardize_data(train_df, std_df)
    # val_df = standardize_data(val_df, std_df)

    train_features = []
    train_labels = []
    for row in train_df.values:
        train_features.append([cell2id[row[0]]])
        train_labels.append([float(row[2])])

    val_features = []
    val_labels = []
    for row in val_df.values:
        val_features.append([cell2id[row[0]]])
        val_labels.append([float(row[2])])

    return train_features, val_features, train_labels, val_labels


def load_pred_data(test_file, cell2id, zscore_method, train_std_file):

    train_std_df = pd.read_csv(
        train_std_file, sep="\t", header=None, names=["dataset", "center", "scale"]
    )
    test_df = pd.read_csv(
        test_file,
        sep="\t",
        header=None,
        names=["cell_line", "smiles", "auc", "dataset"],
    )
    test_std_df = calc_std_vals(test_df, zscore_method)
    for i, row in test_std_df.iterrows():
        dataset = row["dataset"]
        train_entry = train_std_df.query("dataset == @dataset")
        if not train_entry.empty:
            test_std_df.loc[i, "center"] = float(train_entry["center"])
            test_std_df.loc[i, "scale"] = float(train_entry["scale"])
    test_df = standardize_data(test_df, test_std_df)

    feature = []
    label = []
    for row in test_df.values:
        feature.append([cell2id[row[0]]])
        label.append([float(row[2])])
    return feature, label


def prepare_train_data(train_file, cell2id_mapping, zscore_method, std_file):
    train_features, val_features, train_labels, val_labels = load_train_data(
        train_file, cell2id_mapping, zscore_method, std_file
    )
    return (
        torch.Tensor(train_features),
        torch.FloatTensor(train_labels),
        torch.Tensor(val_features),
        torch.FloatTensor(val_labels),
    )


def prepare_predict_data(test_file, cell2id_mapping_file, zscore_method, std_file):
    cell2id_mapping = load_mapping(cell2id_mapping_file, "cell lines")
    test_features, test_labels = load_pred_data(
        test_file, cell2id_mapping, zscore_method, std_file
    )
    return (torch.Tensor(test_features), torch.Tensor(test_labels)), cell2id_mapping


def load_mapping(mapping_file, mapping_type):
    # mapping = {}
    # file_handle = open(mapping_file)
    # for line in file_handle:
    # 	line = line.rstrip().split()
    # 	mapping[line[1]] = int(line[0])

    # file_handle.close()
    # print('Total number of {} = {}'.format(mapping_type, len(mapping)))
    # return mapping
    # Load the file into a DataFrame
    df = pd.read_csv(mapping_file, sep="\s+", header=None, names=["id", "key"])
    # Convert the DataFrame into a dictionary
    mapping = pd.Series(df.id.values, index=df.key).to_dict()
    # Print the total number of items in the mapping
    print("Total number of {} = {}".format(mapping_type, len(mapping)))
    return mapping


def load_mapping_from_list(mapping_file, mapping_type):
    # read text file with pandas and create a dictionary that maps names to ids
    df = pd.read_csv(mapping_file, sep="\s+", header=None, names=["names"])
    mapping = {df["names"][i]: i for i in range(len(df["names"]))}
    print("Total number of {} = {}".format(mapping_type, len(mapping)))
    return mapping


def build_input_vector(input_data, input_features):
    genedim = len(input_features[0, :])
    featdim = len(input_features[0, 0, :])
    feature = np.zeros((input_data.size()[0], genedim, featdim))

    for i in range(input_data.size()[0]):
        feature[i] = input_features[int(input_data[i, 0])]

    feature = torch.from_numpy(feature).float()
    return feature


# build mask: matrix (nrows = number of relevant gene set, ncols = number all genes)
# elements of matrix are 1 if the corresponding gene is one of the relevant genes
def create_term_mask(term_direct_gene_map, gene_dim):
    term_mask_map = {}
    for term, gene_set in term_direct_gene_map.items():
        mask = torch.zeros(len(gene_set), gene_dim)  # .cuda(cuda_id)
        for i, gene_id in enumerate(gene_set):
            mask[i, gene_id] = 1
        term_mask_map[term] = mask
    return term_mask_map


def create_mask_matrix(term_direct_gene_map, gene_dim):
    # creates matrix with dimensions (num_terms, num_genes)
    mask = torch.zeros(len(term_direct_gene_map), gene_dim)  # .cuda(cuda_id)
    for i, gene_set in enumerate(term_direct_gene_map.values()):
        for gene_id in gene_set:
            mask[i, gene_id] = 1
    return mask


def input_connections(feature_to_node_map, node_id_mapping):
    # connect the input feature positions to the node positions in the hidden state
    col = []
    row = []
    for node in feature_to_node_map:  # each systems getting direct input from genes
        for inp_pos in feature_to_node_map[node]:
            col.append(inp_pos)  # from input
            row.append(node_id_mapping[node])  # to node
    return torch.tensor(col, dtype=torch.long), torch.tensor(row, dtype=torch.long)


def get_grad_norm(model_params, norm_type):
    """Gets gradient norm of an iterable of model_params.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Arguments:
            model_params (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                    single Tensor that will have gradients normalized
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                    infinity norm.
    Returns:Total norm of the model_params (viewed as a single vector).
    """
    if isinstance(model_params, torch.Tensor):  # check if parameters are tensorobject
        model_params = [model_params]  # change to list
    model_params = [
        p for p in model_params if p.grad is not None
    ]  # get list of params with grads
    norm_type = float(norm_type)  # make sure norm_type is of type float
    if len(model_params) == 0:  # if no params provided, return tensor of 0
        return torch.tensor(0.0)

    device = model_params[0].grad.device  # get device
    if norm_type == inf:  # infinity norm
        total_norm = max(p.grad.detach().abs().max().to(device) for p in model_params)
    else:  # total norm
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(p.grad.detach(), norm_type).to(device)
                    for p in model_params
                ]
            ),
            norm_type,
        )
    return total_norm


def log_confusion_matrix(module, confmat):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(confmat.cpu().numpy(), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")

    # Convert the matplotlib plot to a PNG image and log it to TensorBoard
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    img = Image.open(buf)
    img = np.array(img)
    # image = plt.imread(buf)
    # loop through loggers and log the image
    module.logger.experiment.add_image(
        "confusion_matrix", img, module.current_epoch, dataformats="HWC"
    )
    # list(module.logger)[0].experiment.add_image('confusion_matrix', img, module.current_epoch, dataformats='HWC')
    # wandb
    # list(module.logger)[1].log_image(key="conf_matrix", images=[img], caption=["confusion_matrix"])
    # for logger in module.trainer.logger:
    # 		if hasattr(logger, "experiment"):
    # 		logger.experiment.add_image('confusion_matrix', img, module.current_epoch, dataformats='HWC')
    # 		else:
    # 			logger.log_image(key="conf_matrix", images=[img], caption=["confusion_matrix"])

    plt.close(fig)


def log_boxplots(module, preds, logits, targets, val="val"):
    """
    Plot box plots of logits and probabilities, grouped by true labels (targets).

    Args:
        logits (torch.Tensor): The predicted logits (unnormalized scores).
        targets (torch.Tensor): The true labels.
    """
    # Convert tensors to numpy for plotting
    logits_np = logits.detach().cpu().numpy()
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot boxplot for logits
    sns.boxplot(x=targets_np, y=logits_np, ax=ax[0])
    ax[0].set_title("Logits Distribution")
    ax[0].set_xlabel("True Label")
    ax[0].set_ylabel("Logits")

    # Plot boxplot for probabilities
    sns.boxplot(x=targets_np, y=preds_np, ax=ax[1])
    ax[1].set_title("Probabilities Distribution")
    ax[1].set_xlabel("True Label")
    ax[1].set_ylabel("Probability")

    # Convert the matplotlib plot to a PNG image and log it to TensorBoard
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    img = Image.open(buf)
    img = np.array(img)
    # image = plt.imread(buf)
    # loop through loggers and log the image
    module.logger.experiment.add_image(
        "box_plot_" + val, img, module.current_epoch, dataformats="HWC"
    )

    plt.close(fig)


def log_scatter(module, preds, targets, val="val"):
    """
    Scatter plot of predictions against targets.
    """
    # Convert tensors to numpy for plotting
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    # Create a figure with two subplots
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot scatter plot
    sns.scatterplot(x=targets_np, y=preds_np, ax=ax)
    ax.set_title("Predictions vs. Targets")
    ax.set_xlabel("True Label")
    ax.set_ylabel("Predicted Label")

    # Convert the matplotlib plot to a PNG image and log it to TensorBoard
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    img = Image.open(buf)
    img = np.array(img)
    # image = plt.imread(buf)
    # loop through loggers and log the image
    module.logger.experiment.add_image(
        "scatterplot_" + val, img, module.current_epoch, dataformats="HWC"
    )

    plt.close(fig)


def load_config(yaml_path):
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config
