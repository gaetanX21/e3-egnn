from torch_geometric.datasets import QM9
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import os
import numpy as np
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
    transform = T.compose([RemoveFields(remove_fields), SetTarget()])
    qm9 = QM9(root=filepath, transform=transform)
    mean, std = qm9.data.y.mean(dim=0), qm9.data.y.std(dim=0)
    qm9.data.y = (qm9.data.y - mean) / std
    return qm9


# def get_chiral_molecules(dataset):
#     """
#     Identify chiral molecules in the QM9 dataset.
    
#     Args:
#         dataset: PyTorch Geometric QM9 dataset.
        
#     Returns:
#         A list of chiral SMILES strings.
#     """
#     chiral_smiles = []
#     for data in dataset:
#         smiles = data.smiles
#         mol = Chem.MolFromSmiles(smiles)
#         if mol is not None:
#             # Find chiral centers
#             chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
#             if chiral_centers:
#                 chiral_smiles.append(smiles)
#     return chiral_smiles


# def preprocess_qm9(dataset, target_properties=["mu", "gap"]):
#     data_list = []
    
#     # Map property names to indices in the QM9 dataset
#     property_indices = {
#         "mu": 0,  # Dipole moment
#         "alpha": 1,  # Polarizability
#         "homo": 2,  # Highest occupied molecular orbital energy
#         "lumo": 3,  # Lowest unoccupied molecular orbital energy
#         "gap": 4,  # Energy gap
#         "r2": 5,  # Electronic spatial extent
#         "zpve": 6,  # Zero-point vibrational energy
#         "u0": 7,  # Internal energy at 0K
#         "u298": 8,  # Internal energy at 298K
#         "h298": 9,  # Enthalpy at 298K
#         "g298": 10,  # Gibbs free energy at 298K
#         "c0": 11,  # Heat capacity at 0K
#         "c298": 12  # Heat capacity at 298K
#     }
    
#     target_indices = [property_indices[prop] for prop in target_properties]
    
#     for i in range(len(dataset)):
#         molecule = dataset[i]
        
#         # Get the node features and edge features (as done before)
#         x = molecule.z.view(-1, 1)  # Node features (atomic numbers)
#         edge_index = molecule.edge_index  # Connectivity (bonds)
#         positions = molecule.pos  # 3D positions of atoms
#         y = torch.tensor([molecule.y[0, idx] for idx in target_indices], dtype=torch.float)
        
#         # Create a PyTorch Geometric Data object for this molecule
#         data = Data(x=x, edge_index=edge_index, pos=positions, y=y)
#         data_list.append(data)
    
#     return data_list

# def generate_enantiomer(molecule):
#     """
#     Generates the mirror image (enantiomer) of a molecule.
#     Returns a new molecule object with reversed 3D coordinates.
#     """
#     mol = Chem.MolFromSmiles(molecule)
#     mol = Chem.AddHs(mol)  # Add hydrogens to the molecule
    
#     # Generate the 3D coordinates for the molecule
#     AllChem.EmbedMolecule(mol)
#     AllChem.UFFOptimizeMolecule(mol)

#     # Get the 3D coordinates of atoms
#     conf = mol.GetConformer()
#     positions = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])

#     # Reflect the coordinates to create the enantiomer
#     reflected_positions = positions * np.array([1, -1, 1])  # Reflect in the Y-axis
#     return reflected_positions


