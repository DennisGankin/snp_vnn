"""
Methods for working with ontologies. 
- create ontology file from networx or cytoscape graph
- Prune ontology file
- Get list of genes from ontology file etc.
"""

import networkx as nx
import json
import pandas as pd
import logging

def from_txt(ontology_file):
    """
    Read ontology file and return a networkx graph.
    """
    # Load the ontology file
    ontology_df = pd.read_csv(ontology_file, sep='\t', header=None, names=['parent', 'child', 'type'])

    # Create a directed graph
    G = nx.from_pandas_edgelist(ontology_df, source='parent', target='child', create_using=nx.DiGraph())
    return G

def get_genes_from_ontology(ontology_file):
    """
    Get list of genes from ontology file.
    """
    # Load the ontology file
    ontology_df = pd.read_csv(ontology_file, sep='\t', header=None, names=['parent', 'child', 'type'])
    # get all rows with type gene
    genes = ontology_df[ontology_df['type'] == 'gene']['child'].to_list()
    return genes

def prune_tree(G):
    """
    Iteratively removes leafes from Graph that are not genes 
    (terms without any connected genes)
    """
    nodes_to_check = list(G.nodes)
    while nodes_to_check:
        node = nodes_to_check.pop()
        # TODO: Check if it is a gene node is dependent on naming 
        if G.out_degree(node) == 0 and G.in_degree(node) > 0 and "NEST" in node: #"GO" in node:  # Check if it is a hanging node
            parents = list(G.predecessors(node))
            G.remove_node(node)
            nodes_to_check.extend(parents)
    return G

def prune_ontology(ontology_file, keep_genes, pruned_ontology_file=None):
    """
    Prune ontology to only include given gene nodes and connected assmeblies.
    """
    # Load the ontology file
    ontology_df = pd.read_csv(ontology_file, sep='\t', header=None, names=['parent', 'child', 'type'])
    
    # Create a directed graph from the ontology data
    G = nx.DiGraph()

    # Add edges to the graph, only add gene if in known genes
    for _, row in ontology_df.iterrows():
        if not (row['type']=='gene' and row['child'] not in keep_genes):
            G.add_edge(row['parent'], row['child'], type=row['type']) 
    
    logging.info(f"Starting with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    # Remove all terms without genes connected
    G = prune_tree(G)
    logging.info(f"Pruned to {len(G.nodes)} nodes and {len(G.edges)} edges.")

    if pruned_ontology_file:
        # Convert the pruned graph back to a dataframe for output
        pruned_edges = [(u, v, G[u][v]['type']) for u, v in G.edges]
        pruned_df = pd.DataFrame(pruned_edges, columns=['parent', 'child', 'type'])
        pruned_df.to_csv(pruned_ontology_file, sep='\t', header=False, index=False)
        logging.info(f"Pruned ontology saved to {pruned_ontology_file}")
        return pruned_df
    else:
        return G

def save_ontology(G, ontology_file):
    """
    Save ontology graph to file.
    """
    edges = [(u, v, G[u][v]['type']) for u, v in G.edges]
    ontology_df = pd.DataFrame(edges, columns=['parent', 'child', 'type'])
    ontology_df.to_csv(ontology_file, sep='\t', header=False, index=False)
    logging.info(f"Ontology saved to {ontology_file}")

def from_nx(cx_path):
    """
    Create ontology from cytoscape network.
    """
    with open(cx_path, 'r') as cx_file:
        cx_data = json.load(cx_file)

    import ndex2
    # Create a NiceCXNetwork object
    network = ndex2.create_nice_cx_from_raw_cx(cx_data)

    # Create a NetworkX graph
    G = nx.DiGraph()

    # Add nodes with attributes
    for node_id, node in network.get_nodes():
        node_attrs = {attr['n']: attr['v'] for attr in network.get_node_attributes(node_id)}
        G.add_node(node_id, n=node['n'].replace(":", "_"), 
                Size=int(node_attrs['Size']),
                Genes=node_attrs['Genes'].split(" "))

    # Add edges with attributes
    for _, edge in network.get_edges():
        G.add_edge(edge['t'], edge['s'])

    # logging basic info
    logging.info(f"Created ontology from {cx_path}")
    logging.info(f"Number of nodes: {G.number_of_nodes()}")
    logging.info(f"Number of edges: {G.number_of_edges()}")

    return G