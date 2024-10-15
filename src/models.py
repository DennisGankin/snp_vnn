import sys
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from . import util


class TermModule(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0):
        super(TermModule, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear1(x)
        x = torch.tanh(x)
        hidden = self.batchnorm(x)
        x = self.linear2(hidden)
        x = torch.tanh(x)

        return x, hidden


class GeneLayer(nn.Module):
    def __init__(self, input_size, out_size, pruning_mask, dropout=0):
        super(GeneLayer, self).__init__()
        # input_size: size of all genes (that's the input diemnsion)
        # out_size: number of all terms connected with genes (that's the output dimension)
        # create linear layer with in and out dimensions
        self.linear = nn.Linear(input_size, out_size)
        # batch norm
        self.batchnorm = nn.BatchNorm1d(out_size)  # maybe not needed
        # prune the layer based on the actual connections (set other weights to zero)
        self.linear = prune.custom_from_mask(self.linear, "weight", pruning_mask)

    def forward(self, x):
        x = self.linear(x)
        x = torch.tanh(x)
        x = self.batchnorm(x)
        return x


class GeneModule(nn.Module):
    def __init__(self, gene_feature_dim, gene_out_dim=1):
        super(GeneModule, self).__init__()
        self.linear = nn.Linear(gene_feature_dim, 1)
        self.batchnorm = nn.BatchNorm1d(1)

    def forward(self, x):
        x = self.linear(x)
        x = torch.tanh(x)
        x = self.batchnorm(x)
        return x


class CompleteGeneModule(nn.Module):
    def __init__(self, num_genes, gene_feature_dim, gene_out_dim=1):
        super(CompleteGeneModule, self).__init__()
        self.linear = LinearColumns(num_genes, gene_feature_dim, 1)
        self.batchnorm = nn.BatchNorm1d(
            1
        )  # might normalizing across gene inputs be better?

    def forward(self, x):
        x = self.linear(x)
        x = torch.tanh(x)
        # permute for batchnorm
        # x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L) , C being channels
        x = x.transpose(2, 1)
        x = self.batchnorm(x)
        x = x.transpose(2, 1)
        # x = x.permute(0, 2, 1)  # (N, C, L) -> (N ,L, C)
        return x


class LinearColumns(nn.Module):
    """
    Module to apply column-specific linear layers on the input tensor.
    Given input of size (batch_size, ... , x1_dim, x2_dim)
    The input has x1_dim columns with x2_dim features each.
    The layer will apply a linear transformation on each column in dimeansion x1 separately.
    Output size will be (batch_size, ... , x1_dim, out_dim)
    """

    def __init__(self, x1_dim, x2_dim, out_dim):
        super(LinearColumns, self).__init__()
        self.x1 = x1_dim
        self.x2 = x2_dim
        self.out_dim = out_dim
        # Trainable weights of size (x1, x2, out_features)
        self.weight = nn.Parameter(torch.randn(x1_dim, x2_dim, out_dim))
        self.bias = nn.Parameter(
            torch.randn(x1_dim, out_dim)
        )  # Bias term of size (x1, out_features)

    def forward(self, x):
        # x is of shape (batch_size, ..., x1, x2)
        *batch_dims, x1, x2 = x.size()
        assert x1 == self.x1 and x2 == self.x2, (
            "Input dimension mismatch. Input:"
            + str(x.size())
            + "Expected: "
            + str(self.x1)
            + " "
            + str(self.x2)
        )

        # Perform element-wise multiplication and then summation along the x2 dimension
        # weights is of shape (x1, x2, out_features)
        # x is of shape (batch_size, ..., x1, x2)
        # We need to sum over the x2 dimension and keep the x1 dimension intact

        # Expand weights to match the batch dimensions
        weights_expanded = self.weight.unsqueeze(0).expand(
            *batch_dims, -1, -1, -1
        )  # Shape: (batch_size, ..., x1, x2, out_features)
        x_expanded = x.unsqueeze(-1).expand(
            *batch_dims, x1, x2, self.out_dim
        )  # Shape: (batch_size, ..., x1, x2, out_features)

        out = (x_expanded * weights_expanded).sum(
            dim=-2
        )  # Summing over x2, resulting in shape: (batch_size, ..., x1, out_features)

        # Add bias
        bias_expanded = self.bias.unsqueeze(0).expand(*batch_dims, x1, self.out_dim)
        out += bias_expanded

        return out


class GenoVNN(nn.Module):
    def __init__(self, args, graph):
        super().__init__()

        self.root = graph.root
        self.num_hiddens_genotype = args.genotype_hiddens

        # dictionary from terms to genes directly annotated with the term
        self.term_direct_gene_map = graph.term_direct_gene_map

        # Dropout Params
        self.min_dropout_layer = args.min_dropout_layer
        self.dropout_fraction = args.dropout_fraction

        # calculate the number of values in a state (term): term_size_map is the number of all genes annotated with the term
        self.cal_term_dim(graph.term_size_map)

        self.gene_id_mapping = graph.gene_id_mapping
        # ngenes, gene_dim are the number of all genes
        self.gene_dim = len(self.gene_id_mapping)

        # No of input features per gene
        self.feature_dim = args.feature_dim

        print("computing masks")
        # self.term_mask = util.create_term_mask(
        #    self.term_direct_gene_map, self.gene_dim
        # )
        self.term_mask_matrix = util.create_mask_matrix(
            self.term_direct_gene_map, self.gene_dim
        )

        self.term_masks = [self.term_mask_matrix[i] == 1 for i in range(len(self.term_direct_gene_map.keys()))]

        self.term_gene_in_map = {} 
        # precompute inputs for each dimension
        #for i, term in enumerate(self.term_direct_gene_map.keys()):
        #    term_gene_out_map[term] = self.term_mask_matrix[i] == 1

        # term to id
        self.term_id_mapping = {term: i for i, term in enumerate(self.term_direct_gene_map.keys())}

        # add modules for neural networks to process genotypes
        print("Constructing first NN layer")
        self.create_gene_layer()
        # self.contruct_direct_gene_layer()
        print("Constructing NN graph")
        self.construct_NN_graph(copy.deepcopy(graph.dG))

        # add module for final layer
        self.add_module(
            "final_aux_linear_layer", nn.Linear(self.num_hiddens_genotype, 1)
        )
        self.add_module("final_linear_layer_output", nn.Linear(1, 1))

    # calculate the number of values in a state (term)
    def cal_term_dim(self, term_size_map):

        self.term_dim_map = {}

        for term, term_size in term_size_map.items():
            num_output = self.num_hiddens_genotype

            # log the number of hidden variables per each term
            num_output = int(num_output)
            self.term_dim_map[term] = num_output

    def create_gene_layer(self):
        # create complete gene module
        gene_layer = CompleteGeneModule(self.gene_dim, self.feature_dim)
        self.add_module("gene_layer", gene_layer)

    # build a layer for forwarding gene that are directly annotated with the term
    def contruct_direct_gene_layer(self):

        # gene modules
        for gene, _ in self.gene_id_mapping.items():
            self.add_module(gene, GeneModule(self.feature_dim))

        # the number of terms that have genes directly connected to them
        terms_with_gene_input = len(self.term_direct_gene_map)
        # create gene_layer
        gene_layer = GeneLayer(
            self.gene_dim, terms_with_gene_input, self.term_mask_matrix
        )

        self.add_module("gene_layer_", gene_layer)

    # start from bottom (leaves), and start building a neural network using the given ontology
    # adding modules --- the modules are not connected yet
    def construct_NN_graph(self, dG):

        self.term_layer_list = []  # term_layer_list stores the built neural network
        self.term_neighbor_map = {}

        # term_neighbor_map records all children of each term
        for term in dG.nodes():
            self.term_neighbor_map[term] = []
            for child in dG.neighbors(term):
                self.term_neighbor_map[term].append(child)

        i = 0
        while True:
            leaves = [n for n in dG.nodes() if dG.out_degree(n) == 0]

            if len(leaves) == 0:
                break

            self.term_layer_list.append(leaves)

            for term in leaves:

                # input size will be #chilren + #genes directly annotated by the term
                input_size = 0

                for child in self.term_neighbor_map[term]:
                    input_size += self.term_dim_map[child]

                if term in self.term_direct_gene_map:
                    input_size += len(self.term_direct_gene_map[term])

                # term_hidden is the number of the hidden variables in each state
                term_hidden = self.term_dim_map[term]

                if i >= self.min_dropout_layer:
                    self.add_module(
                        term, TermModule(input_size, term_hidden, self.dropout_fraction)
                    )
                else:
                    self.add_module(term, TermModule(input_size, term_hidden, 0))

            i += 1
            dG.remove_nodes_from(leaves)

        # Refactored forward method
    def forward_refactored(self, x):
        term_gene_out_map = {}
        hidden_embeddings_map = {}
        aux_out_map = {}

        gene_input = self._modules["gene_layer"](x).squeeze(-1)

        # very sparse though - not entirely correct to multiply!
        #term_gene_out_matrix = torch.matmul(gene_input, self.term_mask_matrix.T)

        for i, term in enumerate(self.term_direct_gene_map.keys()):
            term_gene_out_map[term] = gene_input[:, self.term_masks[i]]

        # Iterate over layers (try to reduce this for loop or parallelize)
        for layer in self.term_layer_list:

            # Iterate over terms in the layer
            for term in layer:
                # Use a list to accumulate child inputs in one operation
                child_input_list = []

                # Instead of iterating, try a vectorized approach if possible
                if len(self.term_neighbor_map[term]) > 0:
                    child_inputs = [hidden_embeddings_map[child] for child in self.term_neighbor_map[term]]
                    child_input_list.extend(child_inputs)

                # If direct gene map exists for this term, append its result
                if term in self.term_direct_gene_map:
                    child_input_list.append(term_gene_out_map[term])

                # If no children, just use the gene input
                if len(child_input_list) == 1:
                    child_input = child_input_list[0]
                else:
                    # Use torch.cat only if multiple inputs need to be concatenated
                    child_input = torch.cat(child_input_list, dim=1)

                # Forward pass through the module for the current term
                x, hidden = self._modules[term](child_input)
                aux_out_map[term], hidden_embeddings_map[term] = x, hidden

        final_input = hidden_embeddings_map[self.root]
        aux_out_map["final_logits"] = self._modules["final_aux_linear_layer"](
            final_input
        )
        aux_out_map["final"] = torch.sigmoid(aux_out_map["final_logits"])

        return aux_out_map, hidden_embeddings_map

    # definition of forward function
    def forward(self, x):

        hidden_embeddings_map = {}
        aux_out_map = {}

        # gene modules
        # for gene, i in self.gene_id_mapping.items():
        # 	feat_out = self._modules[gene](x[:, i, :])
        # 	hidden_embeddings_map[gene] = feat_out #self._modules[gene + '_batchnorm_layer'](feat_out)
        # 	feat_out_list.append(hidden_embeddings_map[gene])
        # gene_input = torch.cat(feat_out_list, dim=1) # needs to be size 64, 584

        gene_input = self._modules["gene_layer"](x).squeeze(-1)

        # gene layer
        # gene_out = self._modules['gene_layer'](gene_input)
        # assign correct part of the output to the correct term

        term_gene_out_map = {}
        # for term, _ in self.term_direct_gene_map.items():
        # 	term_gene_out_map[term] = self._modules[term + '_direct_gene_layer'](gene_input)

        # for i, term in enumerate(self.term_direct_gene_map.keys()):
        # 	term_gene_out_map[term] = gene_out[self.term_mask_matrix[i]==1] # maybe flatten or [:, ...]

        # get input for each term
        for i, term in enumerate(self.term_direct_gene_map.keys()):
            term_gene_out_map[term] = gene_input[:, self.term_mask_matrix[i] == 1]

        for i, layer in enumerate(self.term_layer_list):

            for term in layer:

                child_input_list = []
                for child in self.term_neighbor_map[term]:
                    child_input_list.append(hidden_embeddings_map[child])

                if term in self.term_direct_gene_map:
                    child_input_list.append(term_gene_out_map[term])

                child_input = torch.cat(child_input_list, 1)
                x, hidden = self._modules[term](child_input)
                aux_out_map[term], hidden_embeddings_map[term] = x, hidden

        final_input = hidden_embeddings_map[self.root]
        # aux_layer_out = torch.tanh(self._modules['final_aux_linear_layer'](final_input))
        # aux_out_map['final'] = self._modules['final_linear_layer_output'](aux_layer_out) # this is just a 1 to one with a weight
        # for classification
        aux_out_map["final_logits"] = self._modules["final_aux_linear_layer"](
            final_input
        )
        aux_out_map["final"] = torch.sigmoid(aux_out_map["final_logits"])

        return aux_out_map, hidden_embeddings_map
