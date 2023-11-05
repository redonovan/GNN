import torch

from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dense_to_sparse

@functional_transform('add_missing_edges')
class AddMissingEdges(BaseTransform):
    """Adds all missing edges, except self-loops.
    Adds corresponding missing attributes, as zeros.
    """
    def __call__(self, data):
        # using store lets us obtain num_nodes
        for store in data.edge_stores:
            if store.is_bipartite() or 'edge_index' not in store or 'edge_attr' not in store:
                continue
            # data's info
            edge_index = store.edge_index
            edge_attr  = store.edge_attr
            num_nodes  = store.size(0)
            # dense tensor of all possible edges, minus self loops
            extra_dense = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
            # zero out existing edges
            row, col = edge_index
            extra_dense[row, col] = 0
            # convert to index form
            extra_index, _ = dense_to_sparse(extra_dense)
            # create zero valued attributes
            extra_num  = extra_index.shape[1]
            attr_dim   = edge_attr.shape[1]
            extra_attr = torch.zeros((extra_num, attr_dim))
            # append the extras to the originals
            new_index = torch.cat((edge_index, extra_index), dim=1)
            new_attr  = torch.cat((edge_attr,  extra_attr),  dim=0)
            # and store
            store.edge_index = new_index
            store.edge_attr  = new_attr
            #
        return data

