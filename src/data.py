import torch
import torch_geometric
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from rdkit import Chem
import numpy as np


MAX_ATOMS = 12

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
    

class MaxAtomsFilter(object):
    """
    This transform filters the dataset to only keep molecules that have less than the specified number of atoms.
    """
    def __init__(self, max_atoms):
        self.max_atoms = max_atoms

    def __call__(self, data):
        if data.x.shape[0] <= self.max_atoms:
            return True
        else:
            return False


def load_qm9(filepath="../datasets/"):
    remove_fields = ["z", "smiles", "name", "idx"]
    transform = T.Compose([RemoveFields(remove_fields), SetTarget()])
    filter = MaxAtomsFilter(MAX_ATOMS)
    qm9 = QM9(root=filepath, transform=transform, pre_filter=filter)
    mean, std = qm9.data.y.mean(dim=0), qm9.data.y.std(dim=0)
    qm9.data.y = (qm9.data.y - mean) / std
    return qm9


def split(dataset, train_ratio=0.8, batch_size=64):
    random_idx = torch.randperm(len(dataset))
    n_train = int(len(dataset) * train_ratio)
    n_temp = len(dataset) - n_train
    n_val = n_temp // 2
    train_dataset = dataset[random_idx[:n_train]]
    val_dataset = dataset[random_idx[n_train:n_train+n_val]]
    test_dataset = dataset[random_idx[n_train+n_val:]]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader