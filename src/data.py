from torch_geometric.datasets import QM9
import torch_geometric.transforms as T


class RemoveFields(object):
    """
    This transform removes the specified fields from the data object.
    """
    def __init__(self, fields):
        self.fields = fields

    def __call__(self, data):
        for field in self.fields:
            data.__delattr__(field)
        return data 


class SetTarget(object):
    """
    This transform mofifies the labels vector per data sample to only keep 
    the label for a specific target (there are 19 targets in QM9).

    Note: for this practical, we have hardcoded the target to be target #0,
    i.e. the electric dipole moment of a drug-like molecule.
    (https://en.wikipedia.org/wiki/Electric_dipole_moment)
    """
    def __init__(self, target=0):
        self.target = 0

    def __call__(self, data):
        data.y = data.y[:, self.target]
        return data
    

def load_qm9(filepath="../datasets/"):
    remove_fields = ["z", "smiles", "name", "idx"]
    transform = T.Compose([RemoveFields(remove_fields), SetTarget()])
    qm9 = QM9(root=filepath, transform=transform)
    mean, std = qm9.data.y.mean(dim=0), qm9.data.y.std(dim=0)
    qm9.data.y = (qm9.data.y - mean) / std
    return qm9