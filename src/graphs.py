import sys
import numpy as np
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag


class GeneOntology:
    def __init__(self, gene_id_mapping, ontology_file):
        self.gene_id_mapping = gene_id_mapping
        self.ontology_file = ontology_file
        self.dG = None
        self.root = None
        self.term_size_map = None
        self.term_direct_gene_map = None

        # build 
        self.build_graph(ontology_file)

    def build_graph(self, file_name):
            """
            Load ontology from file
            :param file_name: ontology file
            :return: nx.DiGraph object representing the ontology
            """

            dG = nx.DiGraph()
            term_direct_gene_map = {}
            term_size_map = {}
            gene_set = set()

            file_handle = open(file_name)
            for line in file_handle:
                line = line.rstrip().split()
                if line[2] == "default":
                    dG.add_edge(line[0], line[1])
                else:
                    if line[1] not in self.gene_id_mapping:
                        continue
                    if line[0] not in term_direct_gene_map:
                        term_direct_gene_map[line[0]] = set()
                    term_direct_gene_map[line[0]].add(self.gene_id_mapping[line[1]])
                    gene_set.add(line[1])
            file_handle.close()

            for term in dG.nodes():
                term_gene_set = set()
                if term in term_direct_gene_map:
                    term_gene_set = term_direct_gene_map[term]
                deslist = nxadag.descendants(dG, term)
                for child in deslist:
                    if child in term_direct_gene_map:
                        term_gene_set = term_gene_set | term_direct_gene_map[child]
                # jisoo
                if len(term_gene_set) == 0:
                    print("There is empty terms, please delete term:", term)
                    # dG.remove_node(term)
                    sys.exit(1)
                else:
                    term_size_map[term] = len(term_gene_set)

            roots = [n for n in dG.nodes if dG.in_degree(n) == 0]

            uG = dG.to_undirected()
            connected_subG_list = list(nxacc.connected_components(uG))

            print("There are", len(roots), "roots:", roots[0])
            print("There are", len(dG.nodes()), "terms")
            print("There are", len(connected_subG_list), "connected components")

            if len(roots) > 1:
                print("There are more than 1 root of ontology. Please use only one root.")
                sys.exit(1)
            if len(connected_subG_list) > 1:
                print("There are more than connected components. Please connect them.")
                sys.exit(1)

            self.dG = dG
            self.root = roots[0]
            self.term_size_map = term_size_map
            self.term_direct_gene_map = term_direct_gene_map

            return dG