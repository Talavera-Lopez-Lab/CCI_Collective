import pandas as pd
import numpy as np
from CCI_Collective.utils.matrix_operations import *
from tqdm import tqdm

class cci_collective():

    def __init__(self, adata) -> None:
        self.adata = adata

    def build_features(self, obs_column):
        '''
        returns a table that contains cells as index, obs_column values as columns.
        The Values are the sum of all products of gene expression values of interaction partners between a cell 
        and a cluster of cells as defined by a value in the obs_column (for example a certain cell state)

        obs_column: str = a column contained in the adata.obs table (for example cell_state, batch, etc)
        '''
        if obs_column not in self.adata.obs.columns:
            raise ValueError('obs_column must be a valid column in adata.obs')

        raw_table = pd.read_csv('/mnt/LaCIE/ceger/Projects/CCI_Collective/CCI_Collective/binary_interaction_matrix.csv', index_col=0)
        genes_to_keep = np.intersect1d(self.adata.var_names, raw_table.index)
        filtered_table = raw_table.loc[genes_to_keep, genes_to_keep]
        self.binary_interaction_matrix = filtered_table

        self.filtered_adata = self.adata[:, genes_to_keep]

        clusters = self.adata.obs[obs_column].unique()

        cluster_exp_dict = {}
        for cluster in clusters:
            cluster_exp_dict[cluster] = collapse_matrix(
                self.filtered_adata[self.filtered_adata.obs[obs_column] == cluster].X
            )

        feature_values = []

        for cell in tqdm(self.filtered_adata.obs_names):
            cell_exp_array = self.filtered_adata[self.filtered_adata.obs_names == cell].X.toarray().flatten()

            cell_features = []
            for cluster in clusters:
                cluster_exp_array = cluster_exp_dict[cluster]
                outer_product = np.outer(cell_exp_array, cluster_exp_array.flatten())
                filtered_outer_product = outer_product * self.binary_interaction_matrix.values
                feature_value = filtered_outer_product.sum()
                cell_features.append(feature_value)

            feature_values.append(cell_features)

        feature_table = pd.DataFrame(feature_values, index=self.filtered_adata.obs_names, columns=clusters)

        return feature_table