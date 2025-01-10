import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from tqdm import tqdm
import zipfile
import io
import os
from datetime  import datetime

import torch

import lightning as L
from dataclasses import make_dataclass

from .datasets import UKBSnpLevelDatasetH5OneHot
from .util import load_config
from .graphs import GeneOntology
from .vnn_trainer import GenoVNNLightning, FastVNNLightning, FastVNNLitReg

from captum.attr import (
    LayerConductance,
    LayerIntegratedGradients,
    LayerDeepLift,
    LayerFeatureAblation,
    LayerGradientXActivation,
)

from sklearn.preprocessing import MinMaxScaler

# wrapper model to work with captum attributions
class AnalyzeModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # print(x.shape)
        data = x[:, :-2, :]
        cov = x[:, -2:, 0]
        input = {"x": data, "covariates": cov}
        aux_out_map, _ = self.model(input)
        output = aux_out_map["final"].squeeze(1)
        output_logits = aux_out_map["final_logits"]
        return output_logits

    def training_step(self, batch, batch_idx):
        # data, cov = batch
        # input = {"x": data, "cov": cov}
        return self.model(batch)

class SimpleLayer(L.LightningModule):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)[0]

class SysModelStep1(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for i, layer in enumerate(self.model.model.graph_layer_list[:-1]):
            if i == 0:
                self._modules["layer_" + str(i)] = SimpleLayer(self.model.model._modules["graph_layer_0"])
            else:
                self._modules["layer_" + str(i)] = SimpleLayer(self.model.model._modules["graph_layer_" + str(i + 1)])

    def forward(self, x):

        x = self.model.model._modules["gene_layer"](x).squeeze(-1)

        # loop through layers
        for i, layer in enumerate(self.model.model.graph_layer_list[:-1]): # we don't compute importance for last one
            if i == 0 or i == 1:
                # get the output of the first layer
                x = self._modules["layer_"+str(i)](x)
            else:
                # get the output of the next layer
                x += self._modules["layer_" + str(i)](x)


        #print(sum(sum(x!=0)!=0))
        return x
    
    def training_step(self, batch, batch_idx):
        #data, cov = batch
        #input = {"x": data, "cov": cov}
        return self.model(batch)

class SysModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.systems = SysModelStep1(model)
    def forward(self, x):
        data = x[:,:-2,:]
        cov = x[:,-2:,0]
        final_in = self.systems(data)

        y, hidden = self.model.model._modules["graph_layer_" + str(len(self.model.model.graph_layer_list))](final_in)
        
        final_input = hidden[:, self.model.model.node_id_mapping[self.model.model.root]]

        covariate_out, _ = self.model.model._modules["covariate_module"](cov)
        final_input = torch.cat((final_input, covariate_out), dim=1)

        output = self.model.model._modules["final_aux_linear_layer"](final_input)
        logits = self.model.model._modules["final_linear_layer_output"](output)
        return logits

    def training_step(self, batch, batch_idx):
        #data, cov = batch
        #input = {"x": data, "cov": cov}
        return self.model(batch)


def create_random_input(batch_size=64, input_size=429371):
    x = torch.randint(0, 3, (batch_size, input_size))
    data = x.long()  # Ensure compatibility for one_hot

    # One-hot encode the data (0, 1, or 2) across a new last dimension
    data = torch.nn.functional.one_hot(data, num_classes=3).float()

    baseline = torch.zeros_like(x).long() # do I make the baseline all zeros?????
    baseline_data = torch.nn.functional.one_hot(baseline, num_classes=3).float()
    #baseline_data = torch.zeros_like(data)

    age = torch.randint(1930, 1980, (batch_size, 1, 3))
    sex = torch.randint(0, 2, (batch_size, 1, 3))
    sex_baseline = torch.zeros_like(sex)
    age_baseline = torch.zeros_like(age)
    age_baseline = age_baseline + 1950

    cov = torch.concat([age, sex], dim=1)
    cov_baseline = torch.concat([age_baseline, sex_baseline], dim=1)

    input = torch.concat([data, cov], dim=1)
    input_baseline = torch.concat([baseline_data, cov_baseline], dim=1)

    return input, input_baseline


def run_attributions(attr_func, input, target=0, n_steps=500, batch_size=64, return_convergence_delta=False, baselines=None):
    if type(attr_func) == LayerDeepLift or type(attr_func) == LayerFeatureAblation:
        return attr_func.attribute(input.cuda(), target=target)
    else:
        return attr_func.attribute(
            input.cuda(), 
            target=target,
            baselines=baselines.cuda() if baselines is not None else None,
            n_steps=n_steps, 
            internal_batch_size=batch_size, 
            return_convergence_delta=return_convergence_delta
        )


def compute_attributions(attr_func, n_rounds=10, target=0, n_steps=500, batch_size=64, input_size=429371, return_convergence_delta=False):
    print("Running with n steps:", n_steps)
    input, baselines = create_random_input(batch_size=batch_size, input_size=input_size)    
    if return_convergence_delta:
        attributions_gpu, conv_deltas_gpu = run_attributions(
            attr_func, input, target=target, n_steps=n_steps, batch_size=batch_size, return_convergence_delta=True, baselines=baselines
        )
        print(torch.var(conv_deltas_gpu))
        conv_deltas = conv_deltas_gpu.detach().cpu()
        del conv_deltas_gpu
    else:
        attributions_gpu = run_attributions(
            attr_func, input, target=target, n_steps=n_steps, batch_size=batch_size, baselines=baselines
        )
    attributions = attributions_gpu.detach().cpu()
    del attributions_gpu
    torch.cuda.empty_cache()
    for i in tqdm(range(0, n_rounds - 1)):
        input, baselines = create_random_input(batch_size=batch_size, input_size=input_size)
        if return_convergence_delta:
            attributions_gpu, conv_deltas_gpu = run_attributions(
                attr_func, input, target=target, n_steps=n_steps, batch_size=batch_size, return_convergence_delta=True, baselines=baselines
            )
            print(torch.var(conv_deltas_gpu))
            conv_deltas += conv_deltas_gpu.detach().cpu()
            del conv_deltas_gpu
        else:
            attributions_gpu = run_attributions(
                attr_func, input, target=target, n_steps=n_steps, batch_size=batch_size, baselines=baselines
            )
        attributions += attributions_gpu.detach().cpu()
        del attributions_gpu
        torch.cuda.empty_cache()

    avg_attributions = attributions / n_rounds

    if return_convergence_delta:
        avg_conv_deltas = conv_deltas / n_rounds
        # divide delta by attribution
        #delta_ratio = avg_conv_deltas / avg_attributions
        # print the maximum value and the mean value
        #print("Max delta ratio:", torch.max(delta_ratio).item())
        #print("Mean delta ratio:", torch.mean(delta_ratio).item())

        plt.hist(avg_conv_deltas.numpy().flatten(), bins=50)
        plt.title("Convergence Delta Distribution")
        plt.xlabel("Delta Value")
        plt.ylabel("Frequency")
        plt.show()

        print("Average Convergence Delta:", torch.mean(avg_conv_deltas).item())
        print("Median Convergence Delta:", torch.median(avg_conv_deltas).item())
        print("Variance of Convergence Delta:", torch.var(avg_conv_deltas).item())
        print("Max Convergence Delta:", torch.max(avg_conv_deltas).item())
        print("Min Convergence Delta:", torch.min(avg_conv_deltas).item())

        # Check if deltas are small enough compared to attribution values
        #delta_magnitude_ratio = torch.mean(torch.abs(avg_conv_deltas)) / torch.mean(torch.abs(avg_attributions))
        #print("Delta Magnitude Ratio:", delta_magnitude_ratio.item())
        #if delta_magnitude_ratio < 1e-2:  # Example threshold
        #    print("Convergence deltas are sufficiently small compared to attribution values.")
        #else:
        #    print("Convergence deltas might be significant; investigate further.")

    return avg_attributions

def plot_gwas_gene_overlap(
    attribution_matrices, 
    attribution_methods, 
    gwas_genes, 
    graphs, 
    k=50, 
    abs=True, 
    largest=True
):
    """
    Plots the overlap of GWAS genes included in the top-k genes across models.

    Parameters:
    - attribution_matrices: List of PyTorch tensors (results) from different Methods.
    - attribution_methods: List of names corresponding to each Method.
    - gwas_genes: List or set of GWAS gene names.
    - graphs: List of networkx graphs corresponding to each model.
    - k: Number of top genes to consider.
    - abs: Whether to take the absolute value of attributions before ranking.
    - largest: Whether to consider the largest values as top.

    Returns:
    - None
    """
    # Prepare data for overlap computation
    gwas_gene_sets = {}

    for idx, (results, method_name) in enumerate(zip(attribution_matrices, attribution_methods)):
        # Compute mean attribution scores across samples (dim=0)
        if abs:
            gene_att = results.abs().mean(dim=0)
        else:
            gene_att = results.mean(dim=0)
        
        # Get top-k indices
        topk = torch.topk(gene_att, k, largest=largest)
        
        # Map gene indices to gene names
        node_id_mapping = {node: i for i, node in enumerate(graphs[idx].dG.nodes())}
        node_id_df = pd.DataFrame.from_dict(node_id_mapping, orient="index", columns=["node_id"])
        systems_df = node_id_df.reset_index().set_index("node_id")
        gene_indices = topk.indices.long().numpy()
        gene_names = {systems_df.loc[int(g)]["index"] for g in gene_indices}
        
        # Filter for GWAS genes
        gwas_gene_set = gene_names.intersection(gwas_genes)
        gwas_gene_sets[method_name] = gwas_gene_set

    # Compute overlap counts
    overlap_matrix = pd.DataFrame(index=attribution_methods, columns=attribution_methods, dtype=int)
    
    for i, method_1 in enumerate(attribution_methods):
        for j, method_2 in enumerate(attribution_methods):
            overlap_set = gwas_gene_sets[method_1].intersection(gwas_gene_sets[method_2])
            overlap_count = len(overlap_set)
            print(overlap_set)
            overlap_matrix.loc[method_1, method_2] = overlap_count
            # print unique genes
            print("Unique GWAS genes:", gwas_gene_sets[method_2]-overlap_set)

    print(gwas_gene_sets)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        overlap_matrix.astype(int), 
        annot=True, 
        fmt="d", 
        cmap="coolwarm", 
        cbar_kws={"label": "Number of Overlapping GWAS Genes"}
    )
    plt.title(f"GWAS Gene Overlap Among Top {k} Genes")
    plt.ylabel("Methods")
    plt.xlabel("Methods")
    plt.tight_layout()
    #plt.savefig(f"data/figs/gwas_overlap_heatmap_top_{k}.png")
    plt.show()


def analyze_attribution_methods(
    attribution_matrices,
    attribution_methods,
    gwas_genes,
    graphs,
    k=50,
    largest=True,
    abs=True,
    save=True
):
    """
    Analyzes multiple attribution matrices from different methods.

    Parameters:
    - attribution_matrices: List of PyTorch tensors (results) from different Methods.
    - attribution_methods: List of names corresponding to each Method.
    - gwas_genes: List or set of GWAS gene names.
    - k: Number of top genes to consider for the boxplots.

    Returns:
    - None
    """
    # Assuming systems_df is available in the scope and maps gene indices to gene names
    # If not, you need to pass systems_df as a parameter or adjust accordingly

    current_date=datetime.now().strftime("%Y-%m-%d-%H-%M")

    all_top_genes_data = []
    all_combined_data = []
    percentage_data_top = []
    percentage_data_bottom = []

    for idx, (results, method_name, graph) in enumerate(
        zip(attribution_matrices, attribution_methods, graphs)
    ):
        # Compute mean attribution scores across samples (dim=0)
        if abs == False:
            gene_att = results.mean(dim=0)  # Use abs() if needed
            topk = torch.topk(gene_att, k, largest=largest)
        else:
            gene_att = results.abs().mean(dim=0)  # Use abs() if needed
            topk = torch.topk(gene_att, k)

        # Extract data for top k genes
        data = results[:, topk.indices.long()].detach().numpy()

        # Map gene indices to gene names
        node_id_mapping = {node: i for i, node in enumerate(graph.dG.nodes())}
        node_id_df = pd.DataFrame.from_dict(
            node_id_mapping, orient="index", columns=["node_id"]
        )
        systems_df = node_id_df.reset_index().set_index("node_id")
        gene_indices = topk.indices.long().numpy()
        gene_names = [systems_df.loc[int(g)]["index"] for g in gene_indices]

        # Create a DataFrame for plotting
        df = pd.DataFrame(data, columns=gene_names)

        # Melt the DataFrame for Seaborn
        df_melted = df.melt(var_name="Gene", value_name="Attribution Score")

        # Add GWAS gene status
        df_melted["GWAS Gene"] = df_melted["Gene"].isin(gwas_genes)
        df_melted["Method"] = method_name

        all_top_genes_data.append(df_melted)

        # For violin plot: use all genes
        all_genes_data = pd.DataFrame(
            {
                "Gene": [
                    systems_df.loc[int(g)]["index"] for g in range(results.shape[1])
                ],
                "Attribution Score": np.nan_to_num(gene_att.detach().numpy()) + 1e-29,
                "GWAS Gene": [
                    systems_df.loc[int(g)]["index"] in gwas_genes
                    for g in range(results.shape[1])
                ],
                "Method": method_name,
            }
        )
        all_combined_data.append(all_genes_data)

        # For percentage plot
        # For percentage plots
        method_df = all_genes_data.copy()

        # ---------------------------
        # Percentage in Top N Genes
        # ---------------------------
        method_df_sorted_top = method_df.sort_values(
            by="Attribution Score", ascending=False
        )
        N_values_top = range(10, min(600, len(method_df_sorted_top)) + 1, 10)
        gwas_percentages_top = []
        for N in N_values_top:
            top_N = method_df_sorted_top.head(N)
            count_gwas = top_N["GWAS Gene"].sum()
            gwas_percentages_top.append((count_gwas / N) * 100)
        percentage_df_top = pd.DataFrame(
            {
                "Top N Genes": N_values_top,
                "GWAS Gene Percentage": gwas_percentages_top,
                "Method": method_name,
                "Type": "Top",  # Indicate that this is for top genes
            }
        )
        percentage_data_top.append(percentage_df_top)

        # ---------------------------
        # Percentage in Bottom N Genes
        # ---------------------------
        method_df_sorted_bottom = method_df.sort_values(
            by="Attribution Score", ascending=True
        )
        N_values_bottom = range(10, min(600, len(method_df_sorted_bottom)) + 1, 10)
        gwas_percentages_bottom = []
        for N in N_values_bottom:
            bottom_N = method_df_sorted_bottom.head(N)
            count_gwas = bottom_N["GWAS Gene"].sum()
            gwas_percentages_bottom.append((count_gwas / N) * 100)
        percentage_df_bottom = pd.DataFrame(
            {
                "Top N Genes": N_values_bottom,
                "GWAS Gene Percentage": gwas_percentages_bottom,
                "Method": method_name,
                "Type": "Bottom",  # Indicate that this is for bottom genes
            }
        )
        percentage_data_bottom.append(percentage_df_bottom)

    # Combine data from all methods
    top_genes_df = pd.concat(all_top_genes_data, ignore_index=True)
    combined_df = pd.concat(all_combined_data, ignore_index=True)
    percentage_df_combined = pd.concat(
        percentage_data_top + percentage_data_bottom, ignore_index=True
    )

    print(top_genes_df.head())

    # ----------------------------------
    # Plot 1: Boxplots for Top Genes
    # ----------------------------------
    # Create a FacetGrid with Method as columns
    g = sns.catplot(
        data=top_genes_df,
        kind="box",
        x="Attribution Score",
        y="Gene",
        hue="GWAS Gene",
        col="Method",
        sharey=False,
        sharex=False,
        dodge=False,
        height=15,  # Increase height for better visibility
        aspect=0.3,  # Adjust aspect ratio
        legend="brief",
    )

    # Adjust the plot
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Attribution Score", "Protein")
    # g.add_legend(title='GWAS Gene', loc='upper center')
    sns.move_legend(g, "upper center", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if save:
        plt.savefig("data/figs/gwas_boxplot_all"+current_date+".png")
    else:
        plt.show()
    plt.close()

    # ----------------------------------
    # Plot 2: Violin Plot for GWAS vs Non-GWAS Genes
    # ----------------------------------
    """
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=combined_df,
        x="Method",
        y="Attribution Score",
        hue="GWAS Gene",
        split=True,
        inner="quartile",
        palette="Blues",
        log_scale=True,
    )
    plt.title("Attribution Scores for GWAS vs Non-GWAS Genes")
    plt.ylabel("Attribution Score")
    plt.xlabel("Method")
    plt.legend(title="GWAS Gene")
    plt.tight_layout()
    if save:
        plt.savefig("data/figs/gwas_violin_all"+current_date+".png")
    else:
        plt.show()
    plt.close()
    """

    # ----------------------------------
    # Plot 3: Percentage of GWAS Genes in Top and Bottom N Genes
    # ----------------------------------
    plt.figure(figsize=(12, 6))

    # Separate the data for top and bottom genes
    percentage_top_combined = percentage_df_combined[
        percentage_df_combined["Type"] == "Top"
    ]
    percentage_bottom_combined = percentage_df_combined[
        percentage_df_combined["Type"] == "Bottom"
    ]

    # Plot for Top N Genes
    sns.lineplot(
        data=percentage_top_combined,
        x="Top N Genes",
        y="GWAS Gene Percentage",
        hue="Method",
        style="Type",
        markers=True,
        dashes=False,
        # label='Top N Genes'
    )

    # Plot for Bottom N Genes
    # sns.lineplot(
    #    data=percentage_bottom_combined,
    #    x='Top N Genes',
    #    y='GWAS Gene Percentage',
    #    hue='Method',
    #    style='Type',
    #    markers=True,
    #    dashes=True,
    # label='Bottom N Genes'
    # )

    # Plot overall GWAS rate
    total_gwas_genes = combined_df["GWAS Gene"].sum()
    total_genes = len(combined_df)
    gwas_rate = (total_gwas_genes / total_genes) * 100
    plt.axhline(y=gwas_rate, color="grey", linestyle="--", label="Overall GWAS Rate")

    plt.title("Percentage of GWAS Genes Among n Most Important Genes")
    plt.xlabel("N Most Important Genes")
    plt.ylabel("GWAS Gene Percentage (%)")
    plt.legend(
        title="Method and Type", bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    plt.tight_layout()
    if save:
        plt.savefig("data/figs/gwas_percentage_all"+current_date+".png")
    else:
        plt.show()
    plt.close()

    # ----------------------------------
    # Plot 4: Jaccard index
    # ----------------------------------
    if len(attribution_methods) == 2:
        # Jaccard index computation
        method1_name = attribution_methods[0]
        method2_name = attribution_methods[1]

        # Get sorted DataFrames for the two methods
        method1_df = combined_df[combined_df["Method"] == method1_name]
        method2_df = combined_df[combined_df["Method"] == method2_name]

        method1_sorted = method1_df.sort_values(by="Attribution Score", ascending=False)
        method2_sorted = method2_df.sort_values(by="Attribution Score", ascending=False)

        N_values = range(10, min(600, len(method1_sorted)) + 1, 10)
        jaccard_top = []
        jaccard_gwas = []
        overlap_top_percentage = []
        overlap_gwas_percentage = []

        for N in N_values:
            # Compute Jaccard index for top N genes
            top_N_genes_method1 = set(method1_sorted.head(N)["Gene"])
            top_N_genes_method2 = set(method2_sorted.head(N)["Gene"])
            intersection_top = len(top_N_genes_method1.intersection(top_N_genes_method2))
            union_top = len(top_N_genes_method1.union(top_N_genes_method2))
            jaccard_top.append(intersection_top / union_top if union_top > 0 else 0)
            overlap_top_percentage.append((intersection_top / N) * 100)

            # Compute Jaccard index for GWAS genes among top N genes
            top_N_gwas_method1 = set(
                method1_sorted.head(N).query("`GWAS Gene` == True")["Gene"]
            )
            top_N_gwas_method2 = set(
                method2_sorted.head(N).query("`GWAS Gene` == True")["Gene"]
            )
            intersection_gwas = len(top_N_gwas_method1.intersection(top_N_gwas_method2))
            union_gwas = len(top_N_gwas_method1.union(top_N_gwas_method2))
            jaccard_gwas.append(intersection_gwas / union_gwas if union_gwas > 0 else 0)
            overlap_gwas_percentage.append((intersection_gwas / N) * 100)

        # Create DataFrame for plotting
        jaccard_df = pd.DataFrame(
            {
                "Top N Genes": list(N_values) * 2,
                "Jaccard Index": jaccard_top + jaccard_gwas,
                "Overlap Percentage": overlap_top_percentage + overlap_gwas_percentage,
                "Type": ["Top Genes"] * len(N_values) + ["GWAS Genes"] * len(N_values),
            }
        )

        # Plotting
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot Jaccard Index
        sns.lineplot(
            data=jaccard_df,
            x="Top N Genes",
            y="Jaccard Index",
            hue="Type",
            marker="o",
            ax=ax1,
        )
        ax1.set_ylabel("Jaccard Index")
        ax1.set_xlabel("N Most Important Genes")
        ax1.set_title(f"Jaccard Index and Overlap Percentage Between {method1_name} and {method2_name}")

        # Secondary y-axis for overlap percentages
        ax2 = ax1.twinx()
        sns.lineplot(
            data=jaccard_df,
            x="Top N Genes",
            y="Overlap Percentage",
            hue="Type",
            linestyle="--",
            marker="x",
            ax=ax2,
            legend=False,
        )
        ax2.set_ylabel("Overlap Percentage (%)")

        # Adjust legend
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles=handles, labels=labels, title="Comparison Type")

        plt.tight_layout()
        if save:
            plt.savefig("data/figs/jaccard_overlap_percentage_all" + current_date + ".png")
        else:
            plt.show()
        plt.close()
    return combined_df

def analyze_models():
    sns.set_context("talk")
    print("CUDA available: ", torch.cuda.is_available())
    # load arguments
    print("Loading config...")
    config_file = "src/config/gout_config.yaml"
    config = load_config(config_file)
    config["feature_dim"] = 3

    args = make_dataclass(
        "DataclassFromConfig", [(k, type(v)) for k, v in config.items()]
    )(**config)

    print("Reading BIM and building graph")
    gene_bim_file = "/cluster/project/beltrao/gankin/vnn/data/ukb_gene.bim"
    gene_bim_df = pd.read_csv(gene_bim_file, sep="\t")

    # snp model
    gene_bim_file = "/cluster/project/beltrao/gankin/vnn/geno_vnn/sample/open_targets_bim_sep_t.csv"
    gene_bim_df_snp = pd.read_csv(gene_bim_file, sep="\t")

    snp_id_map_snp = {
        snp: ind
        for snp, ind in zip(
            gene_bim_df_snp["snp"].unique(), range(0, len(gene_bim_df_snp["snp"].unique()))
        )
    }
    snp_id_map = {
        snp: ind
        for snp, ind in zip(
            gene_bim_df["snp"].unique(), range(0, len(gene_bim_df["snp"].unique()))
        )
    }

    graph = GeneOntology(
        snp_id_map,
        "/cluster/project/beltrao/gankin/vnn/snp_vnn/data/NEST_UKB_snp_onto.txt",
        child_node="snp",
    )
    graph_snp = GeneOntology(
        snp_id_map_snp,
        "/cluster/project/beltrao/gankin/vnn/snp_vnn/data/NEST_UKB_OT_snp.txt",
        child_node="snp",
    )

    # load models
    print("Loading models")
    model_names = ["Gout_allvar"]#, "Gout_codingvar"] #"Gout_4", "Gout_6_v0"]
    snp_types = ["all"]#, "coding"]
    model_checkpoints = [
        "data/checkpoints/gout/epoch=2-step=5859.ckpt",
        #"data/checkpoints/gout/epoch=2-step=4578.ckpt",
        #"data/checkpoints/gout/epoch=4-step=7630.ckpt",
        #"data/checkpoints/gout/epoch=6-step=10682.ckpt",
    ] 

    vnn_models = [
    FastVNNLightning.load_from_checkpoint(
        checkpoint, 
        args=args, 
        graph=graph if snp_type == "coding" else graph_snp,
    )
    for checkpoint, snp_type in zip(model_checkpoints, snp_types)
    ]

    # wrap them up
    analyze_models = [AnalyzeModel(model) for model in vnn_models]
    for model in analyze_models:
        model.eval()

    # compute attributions if needed, otherwise load
    att_dir = "data/attributions/gout/"
    # check for each name if file available
    compute = False
    for model_name in model_names:
        if not os.path.exists(f"{att_dir}/{model_name}.pt"):
            compute = True
            break
    if compute:
        print("Computing attributions")
        layer_cond = [
            LayerConductance(model, model.model.model.graph_layer_0.linear2)
            for model in analyze_models
        ]
        attributions = [
            compute_attributions(cond, n_rounds=30, target=0, n_steps=100, batch_size=64, input_size=429371) if snp_type == "coding" else compute_attributions(cond, n_rounds=30, target=0, n_steps=100, batch_size=64, input_size=706556)
            for cond, snp_type in zip(layer_cond,snp_types)
        ]

        # save attributions
        print("Saving attributions")
        # save based on name
        for attr, model_name in zip(attributions, model_names):
            torch.save(attr, f"data/attributions/gout/{model_name}.pt")

    else:
        print("Loading attributions")
        attributions = [
            torch.load(f"{att_dir}/{model_name}.pt")
            for model_name in model_names
        ]

    # compute analysis
    print("Creating plots")
    gwas_df = pd.read_csv(
        "data/gout/gwas-association-downloaded_2024-11-20-EFO_0004274-withChildTraits.tsv",
        sep="\t",
    )

    gwas_genes = set()
    for g in gwas_df.MAPPED_GENE.dropna().apply(lambda x: x.split(" - ")):
        gwas_genes.update(g)

    # reading open targtes
    # Define the file path
    file_path = "data/open_targets/OTAR_gene_dis_assoc_indir_evd.obj.zip"

    # Open the .zip file and extract the pickle file
    with zipfile.ZipFile(file_path, "r") as z:
        # Assuming there's only one file inside the zip, get its name
        pickle_filename = z.namelist()[0]
        with z.open(pickle_filename) as f:
            # Load the pickle object into a pandas DataFrame
            gwas_df = pd.read_pickle(f)

    # Get the gene names
    gout_genes = gwas_df[gwas_df.diseaseName == "gout"].geneSymbol
    # combine with gwas genes
    gwas_genes.update(gout_genes)

    analyze_attribution_methods(
        attributions,
        model_names,
        gwas_genes,
        analyze_models,
        k=50,
        largest=True,
        abs=True,
    )

def analyze_single_model(model_path, config_path, ontology_path, snp_path, att_dir = "data/attributions/gout/"):
    sns.set_context("talk")
    print("CUDA available: ", torch.cuda.is_available())
    # load arguments
    print("Loading config...")
    config = load_config(config_path)
    config["feature_dim"] = 3

    args = make_dataclass(
        "DataclassFromConfig", [(k, type(v)) for k, v in config.items()]
    )(**config)

    print("Reading BIM and building graph")
    gene_bim_df = pd.read_csv(snp_path, sep="\t")

    snp_id_map = {
        snp: ind
        for snp, ind in zip(
            gene_bim_df["snp"].unique(), range(0, len(gene_bim_df["snp"].unique()))
        )
    }

    graph = GeneOntology(
        snp_id_map,
        ontology_path,
        child_node="snp",
    )

    # load model
    print("Loading model")
    model = FastVNNLightning.load_from_checkpoint(
        model_path, 
        args=args, 
        graph=graph,
    )
    model = AnalyzeModel(model)
    model.eval()

    model_name = model_path.split("/")[-1].split(".")[0]

    # compute attributions if needed, otherwise load
    # check for each name if file available
    #if os.path.exists(f"{att_dir}/{model_name}.pt"):
    #    print("Loading attributions")
    #    attributions = torch.load(f"{att_dir}/{model_name}.pt")
    #else:
    print("Computing attributions")
    # calculate input size 
    input_size = len(gene_bim_df["snp"].unique())
    print("Input feature size: ", input_size)
    layer_cond = LayerIntegratedGradients(model, model.model.model.graph_layer_0.linear2, multiply_by_inputs=False)
    attributions = compute_attributions(layer_cond, n_rounds=30, target=0, n_steps=500, batch_size=128, input_size=input_size, return_convergence_delta=True)


    # save attributions
    print("Saving attributions")
    torch.save(attributions, f"{att_dir}/{model_name}_baseline.pt")

    print("Creating plots")
    gwas_genes = get_gwas_genes()
    analyze_attribution_methods(
        [attributions],
        ["Gout_allsnps"],
        gwas_genes,
        [graph],
        k=50,
        largest=True,
        abs=True,
    )

def get_gwas_genes():
    gwas_df = pd.read_csv(
        "data/gout/gwas-association-downloaded_2024-11-20-EFO_0004274-withChildTraits.tsv",
        sep="\t",
    )

    gwas_genes = set()
    for g in gwas_df.MAPPED_GENE.dropna().apply(lambda x: x.split(" - ")):
        gwas_genes.update(g)

    # reading open targtes
    # Define the file path
    file_path = "data/open_targets/OTAR_gene_dis_assoc_indir_evd.obj.zip"

    # Open the .zip file and extract the pickle file
    with zipfile.ZipFile(file_path, "r") as z:
        # Assuming there's only one file inside the zip, get its name
        pickle_filename = z.namelist()[0]
        with z.open(pickle_filename) as f:
            # Load the pickle object into a pandas DataFrame
            gwas_df = pd.read_pickle(f)
    # Get the gene names
    gout_genes = gwas_df[gwas_df.diseaseName == "gout"].geneSymbol
    # combine with gwas genes
    gwas_genes.update(gout_genes)
    return gwas_genes

def attribute_systems(sys_model, input_size=706556):
    cond = LayerConductance(sys_model, sys_model.systems._modules["layer_1"])
    sys_att = compute_attributions(cond, n_rounds=10, target=0, n_steps=1000, batch_size=16, input_size=input_size)
    print("Non zero:", sum(sys_att.mean(dim=0) != 0).numpy())
    for i, layer in enumerate(sys_model.model.model.graph_layer_list[:-3]):
        cond = LayerConductance(sys_model, sys_model.systems._modules["layer_"+str(i+2)])
        sys_att += compute_attributions(cond, n_rounds=10, target=0, n_steps=1000, batch_size=16, input_size=input_size)
        #print("Layer:", i+2)
        print("Layer elems:", len(sys_model.model.model.graph_layer_list[i+2]))
        print("Non zero:", sum(sys_att.mean(dim=0) != 0).numpy())
    
    return sys_att

def analyze_systems(model_path, config_path, ontology_path, snp_path, att_dir = "data/attributions/gout/"):
    sns.set_context("talk")
    print("CUDA available: ", torch.cuda.is_available())
    # load arguments
    print("Loading config...")
    config = load_config(config_path)
    config["feature_dim"] = 3

    args = make_dataclass(
        "DataclassFromConfig", [(k, type(v)) for k, v in config.items()]
    )(**config)

    print("Reading BIM and building graph")
    gene_bim_df = pd.read_csv(snp_path, sep="\t")

    snp_id_map = {
        snp: ind
        for snp, ind in zip(
            gene_bim_df["snp"].unique(), range(0, len(gene_bim_df["snp"].unique()))
        )
    }
    graph = GeneOntology(
        snp_id_map,
        ontology_path,
        child_node="snp",
    )
        # load model
    print("Loading model")
    model = FastVNNLightning.load_from_checkpoint(
        model_path, 
        args=args, 
        graph=graph,
    )
    model = SysModel(model)
    model.eval()

    model_name = model_path.split("/")[-1].split(".")[0]

    print("Computing attributions")
    # calculate input size 
    attributions = attribute_systems(model, input_size=706556)

    print("Saving attributions")
    torch.save(attributions, f"{att_dir}/{model_name}_systems_baseline.pt")

# Function to normalize scores within levels
def normalize_scores(df, levels, score_column="Mean attribution", by_size = False):
    scaler = MinMaxScaler()  # Create a MinMaxScaler instance
    df[score_column+" normalized"] = 0.

    for level in levels:
        # Filter the dataframe for names in the current level
        mask = df['name'].isin(level)
        level_df = df[mask]
        
        # Rescale scores for the current level
        normalized_scores = scaler.fit_transform(level_df[[score_column]])
        if by_size:
            normalized_scores = normalized_scores * len(level_df)

        # Update the scores in the original dataframe
        df.loc[mask, score_column+" normalized"] = normalized_scores.flatten()

    return df

### System analysis
def get_system_df(model, attribution_path):
    # model can be dictionary of paths or actual model instance
    if type(model) == dict:
        print("Loading config...")
        config = load_config(model["config_path"])
        config["feature_dim"] = 3
        args = make_dataclass(
            "DataclassFromConfig", [(k, type(v)) for k, v in config.items()]
        )(**config)

        gene_bim_df = pd.read_csv(model["gene_bim_path"], sep="\t")

        print("SNP map and graph creation")
        snp_id_map = {
            snp: ind
            for snp, ind in zip(
                gene_bim_df["snp"].unique(), range(0, len(gene_bim_df["snp"].unique()))
            )
        }

        graph = GeneOntology(
            snp_id_map,
            model["ontology_path"],
            child_node="snp",
        )

        print("Loading model from checkpoint")
        model = FastVNNLightning.load_from_checkpoint(
            model["model_path"], 
            args=args, 
            graph=graph,
        )

    # get node id mapping and systems
    node_id_df = pd.DataFrame.from_dict(model.model.node_id_mapping, orient="index",columns=["node_id"])
    systems_df = node_id_df.reset_index().set_index("node_id")
    systems_df.columns = ["name"]
    # load system annotations 
    annotation_df = pd.read_csv("data/nest_nodes_annotations.csv")
    annotation_df.name = annotation_df.name.apply(lambda x: "_".join(x.split(":")))
    annotation_df
    # join on name and add annotation to systems df
    systems_df["annotation"] = systems_df.merge(annotation_df, on="name", how="left").Annotation
    # load attributions
    sys_att = torch.load(attribution_path, weights_only=True)

    systems_df["System"] = systems_df["annotation"] + " (" + systems_df["name"] + ")"

    # Compute the absolute mean importance
    systems_df['Mean importance'] = sys_att.abs().mean(dim=0).numpy()

    # Normalize
    return normalize_scores(systems_df, model.model.graph_layer_list, score_column="Mean importance")


# main
if __name__ == "__main__":
    #analyze_models()
    #analyze_single_model(
    #    "data/checkpoints/gout/epoch=2-step=5859.ckpt",
    #    "src/config/gout_config.yaml",
    #    "/cluster/project/beltrao/gankin/vnn/snp_vnn/data/NEST_UKB_OT_snp.txt",
    #    "/cluster/project/beltrao/gankin/vnn/geno_vnn/sample/open_targets_bim_sep_t.csv"
    #    )
    #analyze_systems(
    #    "data/checkpoints/gout/epoch=2-step=4578.ckpt",
    #    "src/config/gout_config.yaml",
    #    "/cluster/project/beltrao/gankin/vnn/snp_vnn/data/NEST_UKB_snp_onto.txt",
    #    "/cluster/project/beltrao/gankin/vnn/data/ukb_gene.bim",
    #    )
    analyze_systems(
        "data/checkpoints/gout/epoch=2-step=5859.ckpt",
        "src/config/gout_config.yaml",
        "/cluster/project/beltrao/gankin/vnn/snp_vnn/data/NEST_UKB_OT_snp.txt",
        "/cluster/project/beltrao/gankin/vnn/geno_vnn/sample/open_targets_bim_sep_t.csv"
        )
