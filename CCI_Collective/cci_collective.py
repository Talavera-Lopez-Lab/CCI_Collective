import pandas as pd
import numpy as np
from CCI_Collective.utils.matrix_operations import *
import CCI_Collective.utils.devices as devices
from tqdm import tqdm
import cupy as cp
import os

class cci_collective():

    def __init__(self, adata) -> None:
        self.adata = adata

    def build_features(
            self,
            obs_column,
            accelerator: str = 'cpu',
            device: int = None
    ):

        '''
        returns a table that contains cells as index, obs_column values as columns.
        The Values are the sum of all products of gene expression values of interaction partners between a cell 
        and a cluster of cells as defined by a value in the obs_column (for example a certain cell state)

        obs_column: str = a column contained in the adata.obs table (for example cell_state, batch, etc)
        accelerator: str = describing the device type you want to use for this operation (cpu or gpu)
        device: int = the id of the GPU in case of multiple available GPUs
        '''

        obs_column_values = self.adata.obs.columns
        accelerator_values = ['cpu', 'gpu']
        devices_info = devices.get_available_devices()
        device_ids = [device['Device ID'] for device in devices_info]
        if obs_column not in obs_column_values:
            raise ValueError('obs_column must be a valid column in adata.obs')
        if accelerator not in accelerator_values:
            raise ValueError(f'accelerator must be one of {accelerator_values}')
        if accelerator != 'cpu':
            if device not in device_ids:
                raise ValueError(f'device must be one of {devices_info}')
            else:
                print(f'selected device {device}')
        binary_interaction_matrix_path = os.path.join(os.path.dirname(__file__), 'binary_interaction_matrix.csv')
        raw_table = pd.read_csv(binary_interaction_matrix_path, index_col=0)
        # Filtering the adata object and interaction matrix for shared genes
        genes_to_keep = np.intersect1d(self.adata.var_names, raw_table.index)
        filtered_table = raw_table.loc[genes_to_keep, genes_to_keep]
        self.binary_interaction_matrix = filtered_table
        self.filtered_adata = self.adata[:, genes_to_keep]

        # getting expression values of cell_clusters
        clusters = self.adata.obs[obs_column].unique()
        cluster_exp_dict = {}
        for cluster in clusters:
            cluster_exp_dict[cluster] = collapse_matrix(
                self.filtered_adata[self.filtered_adata.obs[obs_column] == cluster].X
            )

        if accelerator == 'cpu':
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

        if accelerator != 'cpu':
            with cp.cuda.Device(device):
                feature_values = []
                binary_interaction_matrix_gpu = cp.array(self.binary_interaction_matrix.values)
                for cell in tqdm(self.filtered_adata.obs_names):
                    cell_exp_array = cp.array(self.filtered_adata[self.filtered_adata.obs_names == cell].X.toarray().flatten())
                    cell_features = []
                    for cluster in clusters:
                        cluster_exp_array = cp.array(cluster_exp_dict[cluster].flatten())
                        outer_product = cp.outer(cell_exp_array, cluster_exp_array)
                        filtered_outer_product = outer_product * binary_interaction_matrix_gpu
                        feature_value = filtered_outer_product.sum()
                        cell_features.append(cp.asnumpy(feature_value))
                    feature_values.append(cell_features)


        feature_table = pd.DataFrame(feature_values, index=self.filtered_adata.obs_names, columns=clusters)

        return feature_table